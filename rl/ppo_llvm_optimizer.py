import shutil
import torch
from accelerate import PartialState
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import PPOConfig, PPOTrainer, ModelConfig, ScriptArguments, get_peft_config, get_quantization_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import re
import subprocess
import tempfile
import llvmlite.binding as llvm
import os

llvm.initialize()

def compare_llvm_ir_with_alive2(src_ir: str, tgt_ir: str, alive_tv_path: str, timeout: int = 15) -> dict:
    """Compare two LLVM IR codes using Alive2's alive-tv."""
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "src.ll")
        tgt_file = os.path.join(temp_dir, "tgt.ll")
        with open(src_file, 'w') as f:
            f.write(src_ir)
        with open(tgt_file, 'w') as f:
            f.write(tgt_ir)
        try:
            process = subprocess.Popen(
                [alive_tv_path, src_file, tgt_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=timeout)
            status = "success" if process.returncode == 0 else "failure"
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = "", "Alive2 comparison timed out."
            status = "timeout"
        except Exception as e:
            stdout, stderr = "", str(e)
            status = "error"
        return {"status": status, "stdout": stdout, "stderr": stderr}

def count_instructions(ir_text: str) -> int:
    """Count the number of instructions in LLVM IR."""
    try:
        mod = llvm.parse_assembly(ir_text)
        mod.verify()
        return sum(1 for fn in mod.functions if not fn.is_declaration
                   for bb in fn.blocks for _ in bb.instructions)
    except Exception:
        raise ValueError("Invalid LLVM IR")

LLVM_TAG_RE = re.compile(r"<llvm>(.*?)</llvm>", re.I | re.S)

def extract_llvm_block(reply: str) -> str:
    """Extract text inside <llvm> … </llvm> tags."""
    m = LLVM_TAG_RE.search(reply)
    if not m:
        raise ValueError("No <llvm> … </llvm> block found.")
    return m.group(1).strip()

class CustomRewardFunction:
    def __init__(self, dataset, tokenizer, alive_tv_path):
        self.tokenizer = tokenizer
        self.alive_tv_path = alive_tv_path
        self.mapping = {tuple(example["input_ids"]): (example["non_optimized"], example["LLVM_instcombine"])
                        for example in dataset}

    def __call__(self, query_tensor, response_tensor):
        query_list = tuple(query_tensor.tolist())
        non_optimized, target_optimized = self.mapping[query_list]
        response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)

        try:
            optimized_ir = extract_llvm_block(response_text)
        except ValueError:
            return -1.0  # Penalty for missing <llvm> block

        try:
            num_instructions_opt = count_instructions(optimized_ir)
        except ValueError:
            return -1.0  # Penalty for invalid IR

        num_instructions_non_opt = count_instructions(non_optimized)
        result = compare_llvm_ir_with_alive2(non_optimized, optimized_ir, self.alive_tv_path)

        if result["status"] == "failure":
            reward = -1.0
        elif result["status"] in ["timeout", "error"]:
            reward = -0.5
        else:  # success
            reward = 1.0
            if optimized_ir == target_optimized:
                reward += 0.5
            if optimized_ir == non_optimized:
                reward -= 0.5
            if num_instructions_opt < num_instructions_non_opt:
                reward += 0.2
        return reward

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    alive_tv_path = "/opt/homebrew/bin/alive-tv"


    shutil.rmtree(training_args.output_dir, ignore_errors=True)


    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_args.model_name_or_path = model_name
    torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
                   else getattr(torch, model_args.torch_dtype))
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    policy = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    ref_policy = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs) if not get_peft_config(model_args) else None

    DATASET_DIR = "processed_data_with_prompt"
    ds = load_from_disk(DATASET_DIR)
    eval_samples = 100
    train_dataset = ds.select(range(len(ds) - eval_samples))
    eval_dataset = ds.select(range(len(ds) - eval_samples, len(ds)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        def tokenize(elements):
            outputs = tokenizer(
                elements[dataset_text_field],
                padding=False,
            )
            return {
                "input_ids": outputs["input_ids"],
                "non_optimized": elements["non_optimized"],
                "LLVM_instcombine": elements["LLVM_instcombine"],
            }
        return dataset.map(
            tokenize,
            batched=True,
            num_proc=training_args.dataset_num_proc,
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    training_args.report_to = "wandb"
    training_args.num_sample_generations = 10


    reward_fn = CustomRewardFunction(train_dataset, tokenizer, alive_tv_path)
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=None, 
        value_model=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        reward_fn=reward_fn,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=DATASET_DIR)
    trainer.generate_completions()