from datasets import load_from_disk
import pandas as pd
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import logging
from collections import Counter
from alive2 import compare_llvm_ir_with_alive2
import sys
import re

alive_tv_path = '/home/linuxbrew/.linuxbrew/bin/alive-tv'
dataset_path  = 'processed_data'
model_name    = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
output_file   = 'qwen_1.5b_coder_results.csv'
batch_size    = 1
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

first_batch = not os.path.exists(output_file)

log.info("Loading dataset from %s", dataset_path)
dataset = load_from_disk(dataset_path)
log.info("Dataset contains %d examples", len(dataset))


def strip_fence(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:llvm)?\s*\n?", "", txt, flags=re.IGNORECASE)
    if txt.rstrip().endswith("```"):
        txt = txt.rstrip()[: txt.rstrip().rfind("```")]
    return txt.strip()

def validate_alive2():
    log.info("Validating Alive2 on sample example")
    unopt = dataset[0]['non_optimized']
    opt   = dataset[0]['LLVM_instcombine']
    result = compare_llvm_ir_with_alive2(
        src_ir=unopt, tgt_ir=opt, alive_tv_path=alive_tv_path
    )
    log.info("Alive2 status: %s", result['status'])
    if result.get('stdout'): log.debug("Alive2 stdout:\n%s", result['stdout'])
    if result.get('stderr'): log.debug("Alive2 stderr:\n%s", result['stderr'])
    log.info("Alive2 validation done")

def load_model():
    log.info("Loading model and tokenizer: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log.info("Model and tokenizer loaded")
    return model, tokenizer

validate_alive2()
model, tokenizer = load_model()

system_prompt = (
    "You are an LLVM peephole optimizer. "
    "Pick and apply these transformations: constant-folding, dead-code elimination, "
    "strength reduction, algebraic simplification, or redundant load/store removal.  "
    "Output ONLY the transformed IR, with strictly fewer lines than the input, no fences or comments.  "
    "Before returning, count your output lines and ensure they are at most INPUT_LINES-1."
)

results_counter = Counter()
batch_records    = []

for i in tqdm(range(1000), desc='Optimizing IR'):
    example       = dataset[i]
    unoptimized_ir = example['non_optimized']
    reference_ir   = example['LLVM_instcombine']
    n_lines        = len(unoptimized_ir.splitlines())

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
Now optimize the following LLVM IR (input has {n_lines} lines; output must have at most {n_lines-1} lines):

Unoptimized LLVM IR ({n_lines} lines):
{unoptimized_ir}
"""
    messages = [
        {'role': 'system',  'content': system_prompt.replace("INPUT_LINES", str(n_lines))},
        {'role': 'user',    'content': prompt}
    ]

    log.debug("[Example %d] Sending to model...", i)
    text  = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.7, top_p=0.95)

    gen_ids   = outputs[:, inputs.input_ids.shape[-1]:]
    response  = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    responce = strip_fence(response)

    log.info("[Example %d] Generated %d lines", i, len(response.splitlines()))

    comparison = compare_llvm_ir_with_alive2(src_ir=unoptimized_ir, tgt_ir=response, alive_tv_path=alive_tv_path)
    status     = comparison['status']
    log.info("[Example %d] Alive2 status: %s", i, status)
    results_counter[status] += 1

    rows_after            = len(response.splitlines())
    optimization_performed = (status == 'success') and (rows_after < n_lines)
    exact_match            = (response == reference_ir.strip())

    batch_records.append({
        'unoptimized_ir': unoptimized_ir,
        'response': response,
        'status': status,
        'rows_before': n_lines,
        'rows_after': rows_after,
        'optimization_performed': optimization_performed,
        'exact_match': exact_match,
        'stdout': comparison.get('stdout', ''),
        'stderr': comparison.get('stderr', '')
    })

    if len(batch_records) >= batch_size:
        log.info("Flushing %d records to %s", len(batch_records), output_file)
        df = pd.DataFrame(batch_records)
        df.to_csv(output_file, mode='a', header=first_batch, index=False)
        first_batch    = False
        batch_records  = []

    del inputs, outputs, gen_ids, response, text
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if batch_records:
    log.info("Flushing remaining %d records", len(batch_records))
    pd.DataFrame(batch_records).to_csv(output_file, mode='a', header=first_batch, index=False)

log.info("Done. Results written to %s", output_file)
log.info("Status counts: %s", dict(results_counter))
