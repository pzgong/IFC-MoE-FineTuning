import os

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1"
)


import json
import gc
import time
import random
import resource

import numpy as np
import torch

from pathlib import Path


SEED = 3407


BASE_MODEL_PATH = "Qwen/Qwen3-30B-A3B"
ADAPTER_PATH = "../qlora_weights/qwen3_ifc_qlora"
TEST_JSON = "../datasets/ifc4_semantic_test.json"
OUTPUT_JSON = "../outputs/finetuned_generated_responses.json"
STATISTICS_JSON = "../outputs/finetuned_generation_statistics.json"
CACHE_DIR = "../unsloth_compiled_cache"


MAX_SEQ_LEN = 1024

MAX_NEW_TOKENS = 64

random.seed(SEED)

np.random.seed(SEED)


torch.manual_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


print(
    "Loading libraries..."
)


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


from peft import PeftModel


from tqdm import tqdm

print(
    "Loading test data..."
)



with open(
    TEST_JSON,
    "r",
    encoding="utf-8",
) as f:

    raw_data = json.load(f)



samples = []


for item in raw_data:


    messages = item["messages"]


    system_message = next(
        (
            m["content"]
            for m in messages
            if m["role"] == "system"
        ),
        "",
    )


    user_message = next(
        (
            m["content"]
            for m in messages
            if m["role"] == "user"
        ),
        "",
    )


    reference_message = next(
        (
            m["content"]
            for m in messages
            if m["role"] == "assistant"
        ),
        "",
    )



    samples.append(
        {
            "system": system_message,
            "user": user_message,
            "reference": reference_message,
        }
    )



print(
    f"Total test samples: {len(samples)}"
)

print(
    "Loading tokenizer..."
)


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    padding_side="left",
)



if tokenizer.pad_token_id is None:

    tokenizer.pad_token_id = tokenizer.eos_token_id


print(
    "Loading base model (4-bit NF4)..."
)



bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,

    bnb_4bit_quant_type="nf4",

)



base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL_PATH,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True,

    torch_dtype=torch.bfloat16,

)


print(
    "Loading LoRA adapter..."
)



model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH,

)



model.eval()


DEVICE = model.device



print(
    "Model ready."
)


def build_prompt(
    system: str,
    user: str,
) -> str:


    messages = [

        {
            "role": "system",
            "content": system,
        },

        {
            "role": "user",
            "content": user,
        },

    ]



    return tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,

        enable_thinking=False,

    )

print(
    f"Running warm inference on {len(samples)} samples..."
)



generated_results = []


warm_latencies = []


pbar = tqdm(

    samples,

    total=len(samples),

    desc="Warm inference",

    unit="sample",

    dynamic_ncols=True,

)



for idx, sample in enumerate(pbar):


    prompt = build_prompt(

        sample["system"],

        sample["user"],

    )



    encoded = tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_SEQ_LEN,

    ).to(DEVICE)


    if torch.cuda.is_available():

        torch.cuda.synchronize()



    start_time = time.perf_counter()



    with torch.no_grad():


        outputs = model.generate(

            **encoded,


            max_new_tokens=MAX_NEW_TOKENS,


            do_sample=False,


            pad_token_id=tokenizer.pad_token_id,

        )



    if torch.cuda.is_available():

        torch.cuda.synchronize()



    elapsed = (

        time.perf_counter()

        -

        start_time

    )



    warm_latencies.append(

        elapsed

    )



    input_length = encoded["input_ids"].shape[1]



    generated_tokens = (

        outputs[0][input_length:]

    )


    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    
    
    if "<think>" in generated_text:
    
        generated_text = generated_text.split(
            "</think>"
        )[-1].strip()


    generated_results.append(

        {

            "sample_id": idx + 1,


            "system": sample["system"],


            "user": sample["user"],


            "reference": sample["reference"],


            "generated": generated_text,


            "warm_latency_s": elapsed,

        }

    )

if torch.cuda.is_available():


    peak_vram_gb = (

        torch.cuda.max_memory_allocated()

        /

        1e9

    )


else:


    peak_vram_gb = 0.0


peak_ram_gb = (

    resource.getrusage(

        resource.RUSAGE_SELF

    ).ru_maxrss

    /

    1024**2

)


print(

    f"Saving generated responses to {OUTPUT_JSON}..."

)



os.makedirs(

    os.path.dirname(OUTPUT_JSON),

    exist_ok=True,

)



with open(

    OUTPUT_JSON,

    "w",

    encoding="utf-8",

) as f:


    json.dump(

        generated_results,

        f,

        ensure_ascii=False,

        indent=4,

    )



print(

    "Generated responses saved."

)


warm_latency_mean = float(

    np.mean(warm_latencies)

)


def dir_size_gb(path: str) -> float:


    total_size = 0


    target = Path(path)



    if target.exists():


        for file in target.rglob("*"):


            if file.is_file():


                try:


                    total_size += file.stat().st_size


                except OSError:


                    pass



    return total_size / 1e9




base_model_gb = dir_size_gb(
    BASE_MODEL_PATH
)


adapter_gb = dir_size_gb(
    ADAPTER_PATH
)


cache_gb = dir_size_gb(
    CACHE_DIR
)



full_storage_gb = (

    base_model_gb

    +

    adapter_gb

    +

    cache_gb

)



statistics = {


    "num_samples": len(samples),



    "latency": {


        "warm_run_mean_s_per_sample":

            warm_latency_mean,

    },



    "memory": {


        "peak_vram_gb":

            peak_vram_gb,



        "peak_ram_gb":

            peak_ram_gb,

    },



    "storage": {

    
        "full_storage_gb":
    
            full_storage_gb,
    
    
        "base_model_gb":
    
            base_model_gb,
    
    
        "adapter_gb":
    
            adapter_gb,
    
    
        "cache_gb":
    
            cache_gb,
    
    },

}
with open(

    STATISTICS_JSON,

    "w",

    encoding="utf-8",

) as f:


    json.dump(

        statistics,

        f,

        ensure_ascii=False,

        indent=4,

    )



print(

    "Warm generation statistics saved."

)


print(

    "\n" + "=" * 70

)



print(

    "WARM GENERATION SUMMARY"

)



print(

    "=" * 70

)



print(

    f"Number of samples                 : {len(samples)}"

)



print(

    f"Warm latency (s/sample)           : "

    f"{warm_latency_mean:.4f}"

)


print(

    f"Peak VRAM (GB)                    : "

    f"{peak_vram_gb:.2f}"

)



print(

    f"Peak RAM (GB)                     : "

    f"{peak_ram_gb:.2f}"

)



print(

    f"Full storage (GB)                 : "

    f"{full_storage_gb:.2f}"

)



print(

    f"  Base model storage (GB)          : "

    f"{base_model_gb:.2f}"

)



print(

    f"  LoRA adapter storage (GB)        : "

    f"{adapter_gb:.2f}"

)

print(
    f"  Unsloth compiled cache (GB)       : "
    
    f"{cache_gb:.2f}"
)


print(

    "=" * 70

)



print(

    "Done."

)


gc.collect()



if torch.cuda.is_available():


    torch.cuda.empty_cache()