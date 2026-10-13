import os, gc, sys, logging, torch, pandas as pd, re, random
from collections import Counter
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from alive2 import compare_llvm_ir_with_alive2
from openai import OpenAI
client = OpenAI(api_key = "YOUR_OPENAI_API_KEY")

LOG_DIR      = "logs"
LOG_FILE     = os.path.join(LOG_DIR, "pipeline.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("GPT‑Qwen‑NoFS")

ALIVE_TV_PATH = "/home/linuxbrew/.linuxbrew/bin/alive-tv"
DATASET_PATH  = "processed_data"
OUTPUT_FILE   = "gpt_qwen_nofs_results.csv"

REASONER_MODEL = "gpt-4.1-mini"
CODER_NAME     = "Qwen/Qwen2.5-Coder-3B-Instruct"

USE_MPS = False
DEVICE_MAP   = "mps" if USE_MPS else "auto"
DTYPE        = "auto"
BATCH_SIZE   = 1
NUM_EXAMPLES = 1000

def load_model_and_tokenizer(model_name: str):
    log.info("Loading %s …", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map=DEVICE_MAP
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def chat_openai(model: str, messages, max_tokens=1024, temperature=0.2):
    resp = client.chat.completions.create(
        model       = model,
        messages    = messages
    )
    return resp.choices[0].message.content

def strip_fence(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:llvm)?\s*\n?", "", txt, flags=re.IGNORECASE)
    if txt.rstrip().endswith("```"):
        txt = txt.rstrip()[: txt.rstrip().rfind("```")]
    return txt.strip()

log.info("Loading dataset: %s", DATASET_PATH)
dataset = load_from_disk(DATASET_PATH)
log.info("Dataset size: %d examples", len(dataset))

coder_model, coder_tok = load_model_and_tokenizer(CODER_NAME)
log.info("Coder model ready; reasoner uses %s", REASONER_MODEL)

REASONER_SYSTEM = (
    "You are an LLVM IR optimisation planner.\n"
    "Your ONLY output must be step-by-step reasoning – one numbered bullet per "
    "peephole transformation that will reduce code size while preserving semantics. "
    "Do NOT output any LLVM IR – only the reasoning."
)

CODER_SYSTEM = (
    "You are an advanced LLVM peephole optimiser. Guided strictly by the given "
    "reasoning steps, transform the provided unoptimised LLVM IR so that the "
    "result is semantically equivalent but contains fewer lines.\n"
    "Output ONLY the optimised LLVM IR with NO commentary."
)

first_batch = not os.path.exists(OUTPUT_FILE)
stats, buffer = Counter(), []

for i in tqdm(range(NUM_EXAMPLES), desc="Two‑stage pipeline"):
    ex         = dataset[i]
    unopt_ir   = ex["non_optimized"]
    ref_opt_ir = ex["LLVM_instcombine"]
    n_lines    = len(unopt_ir.splitlines())
    log.info("Example %d | %d lines", i, n_lines)

    reason = chat_openai(
        REASONER_MODEL,
        [
            {"role": "system", "content": REASONER_SYSTEM},
            {"role": "user",
             "content": f"Here is the LLVM IR ({n_lines} lines):\n{unopt_ir}"}
        ]
    ).strip()
    log.debug("Reasoning %d:\n%s", i, reason)

    coder_prompt = (
        f"Reasoning steps:\n{reason}\n\n"
        f"Now optimise the following LLVM IR ({n_lines} lines). "
        f"The result MUST have < {n_lines} lines and MUST NOT contain markdown "
        f"fences (``` or ```llvm). Output the IR verbatim:\n{unopt_ir}"
    )
    coder_msgs = [
        {"role": "system", "content": CODER_SYSTEM},
        {"role": "user",   "content": coder_prompt}
    ]
    templated   = coder_tok.apply_chat_template(
        coder_msgs, tokenize=False, add_generation_prompt=True
    )
    inputs      = coder_tok([templated], return_tensors="pt").to(coder_model.device)
    outputs     = coder_model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.7, top_p=0.95)
    reply_ids   = outputs[:, inputs.input_ids.shape[-1]:]
    optimised   = strip_fence(
        coder_tok.batch_decode(reply_ids, skip_special_tokens=True)[0]
    )
    log.debug("Generated IR %d:\n%s", i, optimised)

    alive2      = compare_llvm_ir_with_alive2(
        src_ir=unopt_ir, tgt_ir=optimised, alive_tv_path=ALIVE_TV_PATH
    )
    status      = alive2["status"]
    rows_after  = len(optimised.splitlines())
    reduced     = status == "success" and rows_after < n_lines
    exact       = optimised.strip() == ref_opt_ir.strip()
    stats[status] += 1

    log.info(
        "idx=%d | status=%s | before=%d after=%d reduced=%s exact=%s",
        i, status, n_lines, rows_after, reduced, exact
    )

    buffer.append({
        "idx": i,
        "reasoning": reason,
        "unoptimised_ir": unopt_ir,
        "generated_ir": optimised,
        "alive2_status": status,
        "rows_before": n_lines,
        "rows_after": rows_after,
        "size_reduced": reduced,
        "exact_match_with_ref": exact,
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

log.info("Done. Status counts: %s", dict(stats))
log.info("Results saved to %s", OUTPUT_FILE)