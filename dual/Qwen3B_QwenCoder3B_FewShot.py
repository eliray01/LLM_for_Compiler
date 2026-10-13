import os, gc, sys, logging, torch, pandas as pd
from collections import Counter
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from alive2 import compare_llvm_ir_with_alive2
import re
import random

LOG_DIR      = "logs"
LOG_FILE     = os.path.join(LOG_DIR, "pipeline.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("QwenPipeline")

ALIVE_TV_PATH = "/home/linuxbrew/.linuxbrew/bin/alive-tv"
DATASET_PATH  = "processed_data"
OUTPUT_FILE   = "qwen_two_stage_results.csv"

REASONER_NAME = "Qwen/Qwen2.5-3B-Instruct"
CODER_NAME    = "Qwen/Qwen2.5-Coder-3B-Instruct"

DEVICE_MAP    = "auto"  
DTYPE         = "auto"
BATCH_SIZE    = 1
NUM_EXAMPLES  = 1000        
FEW_SHOT_K    = 3                     

def load_model_and_tokenizer(model_name: str):
    log.info("Loading %s …", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map=DEVICE_MAP
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_chat(model, tokenizer, messages, max_new_tokens=1024):
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95)
    reply_ids = outputs[:, inputs.input_ids.shape[-1]:]
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

def strip_markdown_fence(text: str) -> str:
    text = re.sub(r"^\s*```(?:llvm)?\s*\n?", "", text, flags=re.IGNORECASE)
    if text.rstrip().endswith("```"):
        text = text.rstrip()[: text.rstrip().rfind("```")]
    return text.strip()

log.info("Loading dataset: %s", DATASET_PATH)
dataset = load_from_disk(DATASET_PATH)
log.info("Dataset size: %d examples", len(dataset))

few_shot_pool = [idx for idx in range(1000, 32000)]
log.info("Few-shot pool: %d short examples", len(few_shot_pool))


reasoner_model, reasoner_tok = load_model_and_tokenizer(REASONER_NAME)
coder_model,    coder_tok    = load_model_and_tokenizer(CODER_NAME)
log.info("Models ready")

REASONER_SYSTEM_PROMPT = (
    "You are an LLVM IR optimisation planner.\n"
    "Your ONLY output must be step-by-step reasoning – one numbered bullet per "
    "peephole transformation that will reduce code size while preserving semantics. "
    "Do NOT output any LLVM IR – only the reasoning."
)

CODER_SYSTEM_PROMPT = (
    "You are an advanced LLVM peephole optimiser. Guided strictly by the given "
    "reasoning steps, transform the provided unoptimised LLVM IR so that the "
    "result is semantically equivalent but contains fewer lines.\n"
    "Output ONLY the optimised LLVM IR with NO commentary."
)

first_batch = not os.path.exists(OUTPUT_FILE)
stats       = Counter()
buffer      = []

for i in tqdm(range(NUM_EXAMPLES), desc="Two-stage pipeline"):
    ex           = dataset[i]
    unopt_ir     = ex["non_optimized"]
    ref_opt_ir   = ex["LLVM_instcombine"]
    n_lines      = len(unopt_ir.splitlines())
    log.info("Example %d | %d lines", i, n_lines)

    reason_msg = [
        {"role": "system", "content": REASONER_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Here is the LLVM IR ({n_lines} lines):\n{unopt_ir}"}
    ]
    reasoning = generate_chat(reasoner_model, reasoner_tok, reason_msg, 1024).strip()
    log.debug("Reasoning for idx %d:\n%s", i, reasoning)


    fs_indices = random.sample(few_shot_pool, FEW_SHOT_K)
    few_shot_blocks = []
    for j, idx in enumerate(fs_indices, 1):
        src = dataset[idx]["non_optimized"]
        tgt = dataset[idx]["LLVM_instcombine"]
        few_shot_blocks.append(
            f"Example {j}:\n\nUnoptimised LLVM IR:\n{src}\n\n"
            f"Optimised LLVM IR:\n{tgt}\n"
        )
    few_shot_text = "\n".join(few_shot_blocks)

    coder_prompt = (
        f"{few_shot_text}\n"
        f"Reasoning steps:\n{reasoning}\n\n"
        f"Now optimise the following LLVM IR ({n_lines} lines). "
        f"The result MUST have < {n_lines} lines and MUST NOT contain markdown "
        f"fences (``` or ```llvm). Output the IR verbatim:\n{unopt_ir}"
    )
    coder_msg = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user",   "content": coder_prompt},
    ]

    raw_ir       = generate_chat(coder_model, coder_tok, coder_msg, 4096).strip()
    optimised_ir = strip_markdown_fence(raw_ir)
    log.debug("Generated IR for idx %d (stripped):\n%s", i, optimised_ir)

    alive2 = compare_llvm_ir_with_alive2(
        src_ir=unopt_ir, tgt_ir=optimised_ir, alive_tv_path=ALIVE_TV_PATH
    )
    status      = alive2["status"]
    rows_after  = len(optimised_ir.splitlines())
    reduced     = status == "success" and rows_after < n_lines
    exact_match = optimised_ir.strip() == ref_opt_ir.strip()
    stats[status] += 1

    log.info(
        "idx=%d | status=%s | before=%d, after=%d, reduced=%s, exact=%s",
        i, status, n_lines, rows_after, reduced, exact_match
    )

    buffer.append({
        "idx": i,
        "reasoning": reasoning,
        "unoptimised_ir": unopt_ir,
        "generated_ir": optimised_ir,
        "alive2_status": status,
        "rows_before": n_lines,
        "rows_after": rows_after,
        "size_reduced": reduced,
        "exact_match_with_ref": exact_match,
        "alive2_stdout": alive2.get("stdout", ""),
        "alive2_stderr": alive2.get("stderr", ""),
    })

    if len(buffer) >= BATCH_SIZE:
        pd.DataFrame(buffer).to_csv(
            OUTPUT_FILE, mode="a", header=first_batch, index=False
        )
        first_batch = False
        buffer.clear()

    gc.collect()
    torch.cuda.empty_cache()
if buffer:
    pd.DataFrame(buffer).to_csv(
        OUTPUT_FILE, mode="a", header=first_batch, index=False
    )

log.info("Pipeline complete. Status counts: %s", dict(stats))
log.info("Results written to %s", OUTPUT_FILE)
