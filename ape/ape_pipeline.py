import os
import re
import json
import random
import tempfile
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, Dataset
from tqdm import tqdm

DATASET_PATH        = "processed_data"
SRC_COL             = "non_optimized"
TGT_COL             = "LLVM_instcombine"
IGNORE_COL          = "prompt"

ALIVE_TV_PATH       = "/home/linuxbrew/.linuxbrew/bin/alive-tv"
OPENAI_MODEL        = "gpt-4o-mini"
OPENAI_API_KEY      = "YOUR_OPENAI_API_KEY"

QWEN_MODEL_NAME     = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
QWEN_MAX_NEW_TOKENS = 4096
QWEN_TEMPERATURE    = 0.6

N_DEMOS          = 10
N_INITIAL_INSTR  = 300
TOP_K            = 20
EVAL_SUBSET_SIZE = 100

RANDOM_SEED         = 42
random.seed(RANDOM_SEED)

LOG_PATH = "ape_pipeline.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
logging.getLogger().addHandler(console)

logging.info("Starting APE pipeline")

logging.info(f"Loading dataset from {DATASET_PATH}")
dataset: Dataset = load_from_disk(DATASET_PATH)
if IGNORE_COL in dataset.column_names:
    dataset = dataset.remove_columns([IGNORE_COL])

logging.info("Splitting dataset: first 1000→test, rest→train (no shuffle)")
test_ds  = dataset.select(range(0, 1000))
train_ds = dataset.select(range(1000, len(dataset)))
logging.info(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

def strip_code_fence(txt: str) -> str:
    txt = re.sub(r"^\s*```[^\n]*\n?", "", txt, flags=re.IGNORECASE)
    if txt.rstrip().endswith("```"):
        txt = txt.rstrip()
        txt = txt[: txt.rfind("```")]
    return txt.strip()

def compare_llvm_ir_with_alive2(src_ir: str, tgt_ir: str,
                                alive_tv_path: str = ALIVE_TV_PATH,
                                timeout: int = 15) -> Dict[str, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "src.ll")
        tgt_file = os.path.join(temp_dir, "tgt.ll")
        with open(src_file, 'w') as f: f.write(src_ir)
        with open(tgt_file, 'w') as f: f.write(tgt_ir)
        try:
            proc = subprocess.Popen(
                [alive_tv_path, src_file, tgt_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            out, err = proc.communicate(timeout=timeout)
            status = "success" if proc.returncode == 0 else "failure"
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = "", "Timeout"
            status = "timeout"
        except Exception as e:
            out, err = "", str(e)
            status = "error"
        return {"status": status, "stdout": out, "stderr": err}

logging.info("Loading Qwen model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
logging.info("Qwen loaded successfully")

def generate_with_qwen(prompt: str) -> str:
    guard = (
        "; IMPORTANT: output ONLY the optimized LLVM IR. "
        "Do NOT include any comments, "
        "no explanations, no markdown fences.\n\n"
    )
    full_prompt = guard + prompt

    logging.info(f"[QWEN] Prompt:\n{full_prompt!r}")

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            temperature=QWEN_TEMPERATURE,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    logging.info(f"[QWEN] Raw generation:\n{text!r}")
    return text[len(full_prompt):].strip()

def openai_chat(messages: List[Dict], n: int = 1, temperature: float = 0.8) -> List[str]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        n=n,
        messages=messages,
    )
    return [c.message.content.strip() for c in resp.choices]

def build_demo_block(ds: Dataset, n: int) -> str:
    exs = ds.select(range(min(n, len(ds)))).to_list()
    parts = []
    for i, ex in enumerate(exs):
        parts.append(
            f"### Example {i+1}\n"
            f"Input IR:\n{ex[SRC_COL]}\n\n"
            f"Output IR:\n{ex.get(TGT_COL, '') or '; (no gold)'}"
        )
    block = "\n\n".join(parts)
    logging.info(f"Built demo block with {len(exs)} examples")
    return block

def propose_instructions(demos: str, n_samples: int) -> List[str]:
    logging.info(f"Requesting {n_samples} instruction candidates from OpenAI")
    sys_msg = (
        "You are an expert compiler engineer. "
        "Infer ONE concise instruction (~1-2 sentences) describing how to transform LLVM IR "
        "from unoptimized to optimized while preserving semantics. "
        "No examples, no meta text."
    )
    user_msg = f"""
I gave a junior engineer an instruction to optimize LLVM IR. They produced these pairs:

{demos}

The instruction they followed was:

<COMPLETE THE INSTRUCTION>
""".strip()

    raw = openai_chat(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": user_msg}],
        n=n_samples,
        temperature=0.8
    )
    uniq = list({r.strip(): None for r in raw}.keys())
    logging.info(f"Received {len(uniq)} unique instructions")
    return uniq

@dataclass
class EvalResult:
    instruction: str
    success_rate: float
    n_success: int
    n_total: int

def eval_one(instruction: str, ex: Dict) -> int:
    prompt = (
        f"{instruction}\n"
        f"### Input IR:\n{ex[SRC_COL]}\n\n"
        f"### Output IR:\n"
    )
    try:
        gen = generate_with_qwen(prompt)
        gen = strip_code_fence(gen)
        logging.info(f"[EVAL] After strip_code_fence:\n{gen!r}")
        res = compare_llvm_ir_with_alive2(ex[SRC_COL], gen)
        logging.info(f"[EVAL] Alive2 status={res['status']}; stdout:\n{res['stdout']!r}\n stderr:\n{res['stderr']!r}")
        return 1 if res["status"] == "success" else 0
    except Exception as e:
        logging.warning(f"eval_one exception: {e}")
        return 0

def score_instruction(instruction: str, pool: Dataset, subset_size: int) -> EvalResult:
    if subset_size > len(pool):
        subset_size = len(pool)
    sample = pool.select(range(subset_size)).to_list()
    succ = sum(eval_one(instruction, ex) for ex in sample)
    rate = succ / subset_size
    logging.info(f"Instruction scored: {rate:.2f} ({succ}/{subset_size})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.debug("torch.cuda.empty_cache() called")
    return EvalResult(instruction, rate, succ, subset_size)

def score_and_select(instructions: List[str], pool: Dataset,
                     subset: int, top_k: int):
    results: List[EvalResult] = []
    logging.info(f"Scoring {len(instructions)} instructions on {subset} examples each")
    for inst in tqdm(instructions, desc="Scoring"):
        results.append(score_instruction(inst, pool, subset))
    results.sort(key=lambda r: r.success_rate, reverse=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.debug("torch.cuda.empty_cache() called after scoring batch")
    logging.info(f"Top-{top_k} instructions selected")
    return results, [r.instruction for r in results[:top_k]]

demos = build_demo_block(train_ds, N_DEMOS)

candidates = propose_instructions(demos, N_INITIAL_INSTR)

all_results, top_instrs = score_and_select(
    candidates, train_ds, EVAL_SUBSET_SIZE, TOP_K
)

logging.info("Top initial instructions:")
for r in all_results[:TOP_K]:
    logging.info(f"{r.success_rate:.2f} | {r.instruction}")

best = top_instrs[0]
logging.info(f"Best instruction chosen: {best!r}")

final = score_instruction(best, test_ds, subset_size=len(test_ds))
logging.info(f"Final test success: {final.success_rate:.3f} ({final.n_success}/{final.n_total})")

with open("instruction_scores.jsonl", "w") as f:
    for r in all_results:
        f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
logging.info("Saved instruction_scores.jsonl")

with open("best_instruction.txt", "w") as f:
    f.write(best)
logging.info("Saved best_instruction.txt")
