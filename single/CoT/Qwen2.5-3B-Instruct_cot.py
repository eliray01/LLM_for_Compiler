from datasets import load_from_disk
import pandas as pd
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from collections import Counter
from alive2 import compare_llvm_ir_with_alive2
import re
import logging
import sys


def strip_fence(txt: str) -> str:
    """Remove leading / trailing markdown fences from a string."""
    txt = re.sub(r"^\s*```(?:llvm)?\s*\n?", "", txt, flags=re.IGNORECASE)
    if txt.rstrip().endswith("```"):
        txt = txt.rstrip()[: txt.rstrip().rfind("```")]
    return txt.strip()


def extract_ir(text: str) -> str:
    """Return only the LLVM‑IR portion of the model output.

    Priority order:
      1. Text enclosed in ```llvm ... ``` fences.
      2. Text after a "### Final IR" (or similar) heading.
      3. Heuristic filter that keeps only lines that *look* like IR.
    """
    m = re.search(r"```(?:llvm)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return strip_fence(m.group(0))
    
    m = re.search(r"###\s*Final\s*IR[\s:]*([\s\S]*)", text, flags=re.IGNORECASE)
    if m:
        inner = m.group(1)
        return strip_fence(inner)

    lines = []
    for line in text.splitlines():
        if re.match(r"\s*(define|declare|;|[@%]|ret |br |store |load )", line):
            lines.append(line)
    return "\n".join(lines).strip()



alive_tv_path = "/home/linuxbrew/.linuxbrew/bin/alive-tv"
dataset_path = "processed_data"
model_name = "Qwen/Qwen2.5-3B-Instruct"
output_file = "qwen_3b_instruct_results_cot.csv"

batch_size = 1

device = "auto"

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "optimizer.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("LLVM-Optimizer")


log.info("Loading dataset from %s", dataset_path)
dataset = load_from_disk(dataset_path)
log.info("Dataset contains %d examples", len(dataset))

log.info("Validating Alive2 on sample example")
self_test = compare_llvm_ir_with_alive2(
    src_ir=dataset[0]["non_optimized"],
    tgt_ir=dataset[0]["LLVM_instcombine"],
    alive_tv_path=alive_tv_path,
)
log.info("Alive2 status: %s", self_test['status'])


def load_model():
    log.info("Loading model and tokenizer: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()
log.info("Model and tokenizer loaded")


system_prompt = (
    "You are an LLVM peephole optimizer. "
    "Pick and apply these transformations: constant‑folding, dead‑code elimination, "
    "strength reduction, algebraic simplification, or redundant load/store removal. "
    "\n\n"
    "You MUST think step‑by‑step. First write your reasoning under the heading '### Plan', "
    "then write the final, optimized LLVM IR under the heading '### Final IR' enclosed in "
    "a fenced code block: ```llvm … ```. "
    "\n\n"
    "The final IR MUST have strictly fewer lines than the input and contain no comments or fences."
)



results_counter = Counter()
first_batch = not os.path.exists(output_file)
batch_records = []

for i in tqdm(range(1000), desc="Evaluating IR optimizations"):
    example = dataset[i]
    unoptimized_ir = example["non_optimized"]
    optimized_ir = example["LLVM_instcombine"]
    n_lines = len(unoptimized_ir.splitlines())
    
    log.info("[Example %d] %d lines input IR", i, n_lines)
    few_shot_examples = f"""
Example 1:

Unoptimized LLVM IR ({len(dataset[30001]['non_optimized'].splitlines())} lines):
{dataset[30001]['non_optimized']}

Optimized LLVM IR ({len(dataset[30001]['LLVM_instcombine'].splitlines())} lines):
{dataset[30001]['LLVM_instcombine']}

Example 2:

Unoptimized LLVM IR({len(dataset[30002]['non_optimized'].splitlines())} lines):
{dataset[30002]['non_optimized']}

Optimized LLVM IR({len(dataset[30002]['LLVM_instcombine'].splitlines())} lines):
{dataset[30002]['LLVM_instcombine']}

Bad Example:

Unoptimized LLVM IR({len(dataset[30003]['non_optimized'].splitlines())} lines):
{dataset[30003]['non_optimized']}

Optimized LLVM IR({len(dataset[30003]['non_optimized'].splitlines())} lines):
{dataset[30003]['non_optimized']}"""

    prompt = f"""{few_shot_examples}
Now optimize the following LLVM IR (input has {n_lines} lines; your *final* IR must have ≤ {n_lines-1} lines):

Unoptimized LLVM IR ({n_lines} lines):
{unoptimized_ir}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    log.debug("[Example %d] Sending to model...", i)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([full_prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=4096, do_sample=True, temperature=0.7, top_p=0.95)

    generated_ids = [out_ids[len(inp_ids):] for inp_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    response_ir = extract_ir(raw_output)

    log.info("[Example %d] Generated %d lines", i, len(response_ir.splitlines()))
    comparison = compare_llvm_ir_with_alive2(
        src_ir=unoptimized_ir,
        tgt_ir=response_ir,
        alive_tv_path=alive_tv_path,
    )
    status = comparison["status"]
    log.info("[Example %d] Alive2 status: %s", i, status)
    results_counter[status] += 1

    rows_before = len(unoptimized_ir.splitlines())
    rows_after = len(response_ir.splitlines())
    optimization_performed = (status == "success") and (rows_after < rows_before)
    exact_match = response_ir.strip() == optimized_ir.strip()

    batch_records.append(
        {
            "unoptimized_ir": unoptimized_ir,
            "response_raw": raw_output,
            "response_ir": response_ir,
            "status": status,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "optimization_performed": optimization_performed,
            "exact_match": exact_match,
            "stdout": comparison.get("stdout", ""),
            "stderr": comparison.get("stderr", ""),
        }
    )

    if len(batch_records) >= batch_size:
        log.info("Flushing %d records to %s", len(batch_records), output_file)
        df_batch = pd.DataFrame(batch_records)
        df_batch.to_csv(output_file, mode="a", header=first_batch, index=False)
        first_batch = False
        batch_records = []

    del model_inputs, generated_ids, raw_output, response_ir, full_prompt
    gc.collect()
    torch.cuda.empty_cache()

if batch_records:
    log.info("Flushing remaining %d records", len(batch_records))
    df_batch = pd.DataFrame(batch_records)
    df_batch.to_csv(output_file, mode="a", header=first_batch, index=False)


log.info("Done. Results written to %s", output_file)
log.info("Status counts: %s", dict(results_counter))
