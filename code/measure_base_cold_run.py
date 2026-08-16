import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SEED = 3407


NUM_COLD_SAMPLES = 100

BASE_MODEL_PATH = "Qwen/Qwen3-30B-A3B"

TEST_JSON = "../datasets/ifc4_semantic_test.json"

STATISTICS_JSON = "../outputs/base_cold_generation_statistics.json"

MAX_SEQ_LEN = 1024


MAX_NEW_TOKENS = 64


def load_sample(sample_index: int) -> dict:

    with open(
        TEST_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        raw_data = json.load(f)


    if not 0 <= sample_index < len(raw_data):
        raise IndexError(
            f"Sample index {sample_index} is outside "
            f"the valid range [0, {len(raw_data) - 1}]."
        )


    messages = raw_data[sample_index]["messages"]


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


    return {
        "system": system_message,
        "user": user_message,
    }

def run_single_cold_worker(
    sample_index: int,
    result_path: str,
) -> None:

    import random
    import time

    import numpy as np
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)


    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    sample = load_sample(
        sample_index
    )

    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",
    )


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )


    model.eval()


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    messages = [
        {
            "role": "system",
            "content": sample["system"],
        },
        {
            "role": "user",
            "content": sample["user"],
        },
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )


    input_device = model.get_input_embeddings().weight.device


    encoded = {
        key: value.to(input_device)
        for key, value in encoded.items()
    }


    input_length = encoded["input_ids"].shape[1]


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    with torch.inference_mode():

        outputs = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()


    generated_tokens = outputs[0][input_length:]


    _ = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    cold_latency_s = (
        time.perf_counter()
        -
        start_time
    )


    result = {
        "cold_latency_s": float(
            cold_latency_s
        )
    }


    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4,
        )


def run_full_cold_benchmark() -> None:

    import numpy as np

    with open(
        TEST_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        raw_data = json.load(f)


    num_samples = len(raw_data)


    if num_samples == 0:
        raise ValueError(
            "The test dataset is empty."
        )


    if NUM_COLD_SAMPLES > num_samples:
        raise ValueError(
            f"NUM_COLD_SAMPLES={NUM_COLD_SAMPLES} exceeds "
            f"the test-set size ({num_samples})."
        )


    rng = np.random.default_rng(SEED)


    sample_indices = rng.choice(
        num_samples,
        size=NUM_COLD_SAMPLES,
        replace=False,
    )


    sample_indices = np.sort(
        sample_indices
    )


    print(
        "\n" + "=" * 72
    )

    print(
        "RANDOM-SAMPLE COLD-START LATENCY BENCHMARK"
    )

    print(
        "=" * 72
    )

    print(
        f"Total test samples     : {num_samples}"
    )

    print(
        f"Sampled cold runs      : {NUM_COLD_SAMPLES}"
    )

    print(
        f"Sampling seed          : {SEED}"
    )

    print(
        f"Model                  : {BASE_MODEL_PATH}"
    )

    print(
        "=" * 72
    )


    cold_latencies = []


    script_path = str(
        Path(__file__).resolve()
    )


    with tempfile.TemporaryDirectory(
        prefix="qwen3_cold_benchmark_"
    ) as temp_dir:


        temp_dir = Path(temp_dir)


        for run_index, sample_index in enumerate(
            sample_indices,
            start=1,
        ):


            sample_index = int(
                sample_index
            )


            sample_id = (
                sample_index + 1
            )


            result_path = (
                temp_dir
                /
                f"sample_{sample_id:05d}.json"
            )


            print(
                f"Cold-start run "
                f"{run_index}/{NUM_COLD_SAMPLES} "
                f"(test sample {sample_id})"
            )


            command = [
                sys.executable,
                script_path,
                "--worker",
                "--sample-index",
                str(sample_index),
                "--result-path",
                str(result_path),
            ]


            try:

                subprocess.run(
                    command,
                    check=True,
                )


            except subprocess.CalledProcessError as exc:

                raise RuntimeError(
                    f"Cold-start benchmark failed "
                    f"for sample {sample_id}."
                ) from exc


            if not result_path.exists():

                raise RuntimeError(
                    f"Worker result was not created "
                    f"for sample {sample_id}."
                )


            with open(
                result_path,
                "r",
                encoding="utf-8",
            ) as f:

                result = json.load(f)


            if "cold_latency_s" not in result:

                raise RuntimeError(
                    f"Invalid worker result "
                    f"for sample {sample_id}."
                )


            latency = float(
                result["cold_latency_s"]
            )


            if (
                not np.isfinite(latency)
                or latency <= 0
            ):

                raise RuntimeError(
                    f"Invalid cold latency "
                    f"for sample {sample_id}: "
                    f"{latency}"
                )


            cold_latencies.append(
                latency
            )


            print(
                f"Latency: {latency:.4f} s"
            )


    latency_array = np.asarray(
        cold_latencies,
        dtype=np.float64,
    )


    end_to_end_cold_run_mean_s_per_sample = float(
        np.mean(
            latency_array
        )
    )


    cold_latency_p50_s_per_sample = float(
        np.percentile(
            latency_array,
            50,
            method="linear",
        )
    )


    cold_latency_p90_s_per_sample = float(
        np.percentile(
            latency_array,
            90,
            method="linear",
        )
    )


    statistics = {

        "end_to_end_cold_run_mean_s_per_sample":
            end_to_end_cold_run_mean_s_per_sample,

        "cold_latency_p50_s_per_sample":
            cold_latency_p50_s_per_sample,

        "cold_latency_p90_s_per_sample":
            cold_latency_p90_s_per_sample,

    }


    statistics_path = Path(
        STATISTICS_JSON
    )


    statistics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        statistics_path,
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
        "\n" + "=" * 72
    )

    print(
        "COLD-START LATENCY SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        "End-to-end latency cold run (s/sample) : "
        f"{end_to_end_cold_run_mean_s_per_sample:.4f}"
    )


    print(
        "Cold latency p50 (s/sample)             : "
        f"{cold_latency_p50_s_per_sample:.4f}"
    )


    print(
        "Cold latency p90 (s/sample)             : "
        f"{cold_latency_p90_s_per_sample:.4f}"
    )


    print(
        "=" * 72
    )


    print(
        f"Statistics saved to: "
        f"{STATISTICS_JSON}"
    )


def main() -> None:


    parser = argparse.ArgumentParser(
        description=(
            "Random-sample process-level cold-start "
            "latency benchmark for Qwen3-30B-A3B."
        )
    )


    parser.add_argument(
        "--worker",
        action="store_true",
        help=(
            "Run one cold-start worker. "
            "Used internally by the parent process."
        ),
    )


    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--result-path",
        type=str,
        default="",
    )


    args = parser.parse_args()


    if args.worker:


        if not args.result_path:
            raise ValueError(
                "--result-path is required "
                "in worker mode."
            )


        run_single_cold_worker(
            sample_index=args.sample_index,
            result_path=args.result_path,
        )


    else:


        run_full_cold_benchmark()

if __name__ == "__main__":
    main()