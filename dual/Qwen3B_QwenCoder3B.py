import os, gc, torch, pandas as pd
from collections import Counter
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from alive2 import compare_llvm_ir_with_alive2
import re

ALIVE_TV_PATH   = '/home/linuxbrew/.linuxbrew/bin/alive-tv'
DATASET_PATH    = 'processed_data'
OUTPUT_FILE     = 'qwen_two_stage_results.csv'

REASONER_NAME   = "Qwen/Qwen2.5-3B-Instruct"
CODER_NAME      = "Qwen/Qwen2.5-Coder-3B-Instruct"

DEVICE_MAP      = 'auto'     
DTYPE           = 'auto'      
BATCH_SIZE      = 1
NUM_EXAMPLES    = 1000



def strip_fence(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:llvm)?\s*\n?", "", txt, flags=re.IGNORECASE)
    if txt.rstrip().endswith("```"):
        txt = txt.rstrip()[: txt.rstrip().rfind("```")]
    return txt.strip()

def load_model_and_tokenizer(model_name: str):
    print(f"▸ Loading {model_name} …")
    model      = AutoModelForCausalLM.from_pretrained(
                     model_name, torch_dtype=DTYPE, device_map=DEVICE_MAP
                 )
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_chat(model, tokenizer, messages, max_new_tokens=1024):
    chat_text   = tokenizer.apply_chat_template(
                      messages, tokenize=False, add_generation_prompt=True
                  )
    model_in    = tokenizer([chat_text], return_tensors='pt').to(model.device)
    out_ids     = model.generate(**model_in, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95)

    gen_ids     = out_ids[:, model_in.input_ids.shape[-1]:]
    reply       = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return reply


print("Loading dataset …")
ds        = load_from_disk(DATASET_PATH)
print(f"  ✓ {len(ds)} usable examples.")


reasoner_model, reasoner_tok = load_model_and_tokenizer(REASONER_NAME)
coder_model,    coder_tok    = load_model_and_tokenizer(CODER_NAME)
print("Both models loaded.\n")


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

first_batch     = not os.path.exists(OUTPUT_FILE)
stats           = Counter()
buffer          = []

for i in tqdm(range(NUM_EXAMPLES), desc="Evaluating two-stage pipeline"):
    ex              = ds[i]
    unopt_ir        = ex['non_optimized']
    ref_opt_ir      = ex['LLVM_instcombine']
    n_lines         = len(unopt_ir.splitlines())

    reasoner_msg = [
        {'role': 'system', 'content': REASONER_SYSTEM_PROMPT},
        {'role': 'user',   'content': f"Here is the LLVM IR ({n_lines} lines):\n{unopt_ir}"}
    ]
    reasoning_out = generate_chat(
        model=reasoner_model, tokenizer=reasoner_tok,
        messages=reasoner_msg, max_new_tokens=1024
    ).strip()

    coder_user_prompt = (
        f"Reasoning steps:\n{reasoning_out}\n\n"
        f"Now apply those steps to the following LLVM IR ({n_lines} lines). "
        f"The result MUST have < {n_lines} lines:\n{unopt_ir}"
    )

    coder_msg = [
        {'role': 'system', 'content': CODER_SYSTEM_PROMPT},
        {'role': 'user',   'content': coder_user_prompt}
    ]
    optimised_ir = generate_chat(
        model=coder_model, tokenizer=coder_tok,
        messages=coder_msg, max_new_tokens=4096
    ).strip()

    optimised_ir = strip_fence(optimised_ir)

    alive2_res = compare_llvm_ir_with_alive2(
        src_ir      = unopt_ir,
        tgt_ir      = optimised_ir,
        alive_tv_path = ALIVE_TV_PATH
    )
    status      = alive2_res['status']
    rows_before = n_lines
    rows_after  = len(optimised_ir.splitlines())
    optimised   = (status == 'success') and (rows_after < rows_before)
    exact_eq    = optimised_ir.strip() == ref_opt_ir.strip()

    stats[status] += 1

    buffer.append({
        'idx'                  : i,
        'reasoning'            : reasoning_out,
        'unoptimised_ir'       : unopt_ir,
        'generated_ir'         : optimised_ir,
        'alive2_status'        : status,
        'rows_before'          : rows_before,
        'rows_after'           : rows_after,
        'size_reduced'         : optimised,
        'exact_match_with_ref' : exact_eq,
        'alive2_stdout'        : alive2_res.get('stdout', ''),
        'alive2_stderr'        : alive2_res.get('stderr', ''),
    })

    if len(buffer) >= BATCH_SIZE:
        pd.DataFrame(buffer).to_csv(
            OUTPUT_FILE, mode='a', header=first_batch, index=False
        )
        first_batch = False
        buffer.clear()

    gc.collect()
    torch.cuda.empty_cache()

if buffer:
    pd.DataFrame(buffer).to_csv(
        OUTPUT_FILE, mode='a', header=first_batch, index=False
    )


print("\nPipeline complete.")
print("Alive2 status counts:", dict(stats))
print("Results written to", OUTPUT_FILE)
