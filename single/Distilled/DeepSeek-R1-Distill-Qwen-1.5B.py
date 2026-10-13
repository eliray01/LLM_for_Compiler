import os, sys, gc, re, logging, random, torch, pandas as pd
from pathlib import Path
from collections import Counter
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from alive2 import compare_llvm_ir_with_alive2
import llvmlite.binding as llvm

llvm.initialize()

MODEL_NAME     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH   = "processed_data"
RESULTS_CSV    = "deepseek_results.csv"
ALIVE_TV_PATH  = "/home/linuxbrew/.linuxbrew/bin/alive-tv"

DEVICE_MAP     = "auto"
DTYPE          = torch.float16
NUM_EXAMPLES   = 1000
MAX_NEW_TOKENS = 8196
TEMPERATURE    = 0.6
BATCH_SIZE     = 1 

LOG_DIR        = "logs"
LOG_FILE       = Path(LOG_DIR) / "deepseek_pipeline.log"
Path(LOG_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("DeepSeekPipeline")


def count_instructions(ir_text: str) -> int:
    mod = llvm.parse_assembly(ir_text)
    mod.verify()
    return sum(
        1
        for fn in mod.functions
        if not fn.is_declaration
        for bb in fn.blocks
        for _ in bb.instructions
    )

def strip_ir_metadata(ir: str) -> str:
    filtered_lines = []
    for line in ir.splitlines():
        if line.startswith('attributes '):
            continue
        if re.match(r'^\s*!', line):
            continue
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def strip_fence(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:llvm)?\s*\n?", "", txt, flags=re.IGNORECASE)
    if txt.rstrip().endswith("```"):
        txt = txt.rstrip()[: txt.rstrip().rfind("```")]
    return txt.strip()

HEADER_PATTERN = re.compile(
    r"((?:;\s*ModuleID\s*=.*\n"
    r"source_filename\s*=.*\n"
    r"target\s+datalayout\s*=.*\n"
    r"target\s+triple\s*=.*\n)\n?)",
    re.I
)

def split_ir_header(ir_text: str):
    m = HEADER_PATTERN.match(ir_text)
    if not m:
        raise ValueError("Expected IR header not found.")
    header = m.group(1)
    body   = ir_text[m.end():].lstrip("\n")
    return header, body

def rebuild_ir(header: str, optimised_body: str) -> str:
    return f"{header.rstrip()}\n\n{optimised_body.lstrip()}"

LLVM_TAG_RE = re.compile(r"<llvm>(.*?)</llvm>", re.I | re.S)

def extract_llvm_block(reply: str) -> str:
    m = LLVM_TAG_RE.search(reply)
    if not m:
        raise ValueError("No <llvm> … </llvm> block found in model output.")
    return m.group(1).strip()

log.info("Loading model %s …", MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE_MAP)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
log.info("Model & tokenizer ready.")

log.info("Loading dataset from %s …", DATASET_PATH)
dataset = load_from_disk(DATASET_PATH)
if NUM_EXAMPLES is None:
    NUM_EXAMPLES = len(dataset)
log.info("Dataset size: %d examples (processing %d)", len(dataset), NUM_EXAMPLES)

IR_RULES = r"""
You are an LLVM IR peephole optimizer.
Input: just the module body (everything after the `target datalayout` / `target triple` lines).
Output: a body that
  • assembles under `llvm-as && opt -verify -S`,
  • passes Alive2 (`alive-tv original.ll optimised.ll`),
  • and has strictly fewer static instructions than the input.

0. **Must assemble.** Do not change `define` vs. `declare`, braces, or syntax.  
   If you cannot emit valid IR, return the original body verbatim.

1. **Preserve definitions.**  
   Every function received as `define … { … }` must stay a `define`.  
   Only emit `declare` for symbols not defined in the input.

2. **Decls for intrinsics.**  
   If you add or keep any external/intrinsic symbol, emit a matching `declare`.  
   Remove any now-unused `declare`s.

3. **Header untouched.**  
   Do not output or alter the module header—touch only the body.

4. **Block labels.**  
   Label every basic block with a human name (e.g. `entry`, `loop.latch`);  
   never numeric-only labels like `0:`.

5. **SSA renaming.**  
   Rename SSA values to `%0`, `%1`, … with no gaps within each function.

6. **Function attrs.**  
   Place attributes after `define … @name(params)` using only LLVM 14–valid names  
   (e.g. `nounwind`, `noundef`, `uwtable`).

7. **Comments.**  
   No comment may contain `%` or basic-block labels.

8. **Memory & UB safety.**  
   Do not introduce undefined behavior (`undef`, `poison`, etc.).  
   Reorder loads/stores only when provably independent.

9. **Loop soundness.**  
   Apply only peephole rewrites that are correct under Alive2’s single-loop unroll.

10. **Size reduction.**  
    The optimized body must have strictly fewer static instructions.  
    If you cannot prove a safe reduction, output the original body unchanged.

11. **Output format – reply must be exactly:**  
        <think>  
        …your internal reasoning…  
        </think>  
        <llvm>  
        …optimised IR body…  
        </llvm>
    No markdown, explanations, or extra text outside those blocks.
"""


def generate_reply(unopt_body: str, n_lines: int) -> str:
    prompt = (
        f"{IR_RULES}\n\n"
        "General directives:\n"
        "- Optimise the IR **body** given below.  Do not emit ModuleID, "
        "source_filename, target datalayout, or target triple.\n"
        "- Enclose the optimised body in <llvm> … </llvm> tags.\n\n"
        f"The body to optimise:\n\n{unopt_body}"
    )
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id,
    )
    reply_ids = outputs[:, inputs.input_ids.shape[-1] :]
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

first_batch = not Path(RESULTS_CSV).exists()
results_buf = []
status_counts = Counter()

for idx in tqdm(range(NUM_EXAMPLES), desc="DeepSeek pipeline"):
    unopt_ir = dataset[idx]["non_optimized"]

    unopt_ir = strip_ir_metadata(unopt_ir)
    header, unopt_body = split_ir_header(unopt_ir)
    n_lines_before = unopt_body.count("\n") + 1

    try:
        insns_before = count_instructions(unopt_body)
    except Exception as e:
        log.error("Failed to count instructions before optimisation (idx=%d): %s", idx, e)
        insns_before = None

    raw_reply = generate_reply(unopt_body, n_lines_before)
    try:
        optimised_body = extract_llvm_block(raw_reply)
    except ValueError as e:
        log.warning("idx=%d – missing <llvm> tags, skipping.\nRaw reply:\n%s", idx, raw_reply)
        status = "no_llvm_block"
        status_counts[status] += 1
        continue
    optimised_body = optimised_body.strip()
    optimised_body_no_fence = strip_fence(optimised_body)
    optimised_body_no_md = strip_ir_metadata(optimised_body_no_fence)

    alive2 = compare_llvm_ir_with_alive2(
        src_ir=unopt_body,
        tgt_ir=optimised_body_no_md,
        alive_tv_path=ALIVE_TV_PATH,
    )
    status = alive2.get("status", "error")
    status_counts[status] += 1

    insns_after = None
    if status == "success":
        try:
            insns_after = count_instructions(optimised_body_no_md)
        except Exception as e:
            log.error("idx=%d – instruction count after optimisation failed: %s", idx, e)

    log.info(
        "idx=%d | status=%s | insns_before=%s | insns_after=%s",
        idx,
        status,
        insns_before,
        insns_after if insns_after is not None else "–",
    )

    results_buf.append(
        {
            "idx": idx,
            "instructions_before": insns_before,
            "instructions_after": insns_after,
            "alive2_status": status,
            "alive2_stdout": alive2.get("stdout", ""),
            "alive2_stderr": alive2.get("stderr", ""),
            "unoptimised_ir": unopt_body,
            "generated_ir": optimised_body_no_md,
        }
    )

    if len(results_buf) >= BATCH_SIZE:
        pd.DataFrame(results_buf).to_csv(
            RESULTS_CSV,
            mode="a",
            header=first_batch,
            index=False,
        )
        first_batch = False
        results_buf.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if results_buf:
    pd.DataFrame(results_buf).to_csv(
        RESULTS_CSV, mode="a", header=first_batch, index=False
    )

log.info("Pipeline complete – status counts: %s", dict(status_counts))
log.info("Detailed results written to %s", RESULTS_CSV)
