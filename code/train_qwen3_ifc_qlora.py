import os
import json
import time
import random

from pathlib import Path

import numpy as np
import torch
import transformers
import trl
import unsloth

from unsloth import FastLanguageModel

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

SEED = 3407


random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)


MODEL_PATH = "Qwen/Qwen3-30B-A3B"

STAGE1_DATA = "../datasets/ifc4_semantic_train.json"

STAGE2_DATA = "../datasets/ifc4_operational_train.json"

OUTPUT_DIR = "../qlora_weights/qwen3_ifc_qlora"


MAX_SEQ_LENGTH = 7168

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0

LEARNING_RATE = 1e-4

PER_DEVICE_BATCH_SIZE = 2

GRAD_ACCUMULATION = 16

MAX_STEPS = 500


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



def slim_tools(tools):

    if not isinstance(tools, list):

        return tools


    new_tools = []


    for tool in tools:


        if not isinstance(tool, dict):

            continue



        if tool.get("type") != "function":

            new_tools.append(tool)

            continue



        fn = tool.get(
            "function",
            {}
        )


        params = fn.get(
            "parameters",
            {}
        )


        if isinstance(params, dict):

            params = dict(params)


            params.pop(
                "title",
                None
            )

            params.pop(
                "$defs",
                None
            )



        new_tools.append(
            {
                "type": "function",

                "function":
                {
                    "name": fn.get("name"),

                    "parameters": params

                }
            }
        )


    return new_tools




def normalize_sample(sample):

    if "messages" not in sample:

        raise ValueError(
            "Each sample must contain a 'messages' field."
        )


    messages = [

        dict(m)

        for m in sample["messages"]

    ]



    if "tools" in sample:


        if len(messages) == 0:

            raise ValueError(
                "Tool samples require non-empty messages."
            )



        tools = slim_tools(
            sample["tools"]
        )


        tools_json = json.dumps(

            tools,

            ensure_ascii=False,

            separators=(
                ",",
                ":"
            )

        )


        if messages[0]["role"] == "system":


            messages[0]["content"] = (

                "TOOLS_JSON:"

                +

                tools_json

                +

                "\n\n"

                +

                messages[0]["content"]

            )


    return {

        "messages": messages

    }




def build_dataset(paths):

    samples = []


    for path in paths:


        data = load_json(path)


        for sample in data:


            samples.append(

                normalize_sample(sample)

            )



    return Dataset.from_list(samples)


def main():


    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )


    model, tokenizer = FastLanguageModel.from_pretrained(

        model_name=MODEL_PATH,
    
        max_seq_length=MAX_SEQ_LENGTH,
    
        dtype=torch.bfloat16,
    
        load_in_4bit=True
    
    )



    model = FastLanguageModel.get_peft_model(

        model,


        r=LORA_R,


        lora_alpha=LORA_ALPHA,


        lora_dropout=LORA_DROPOUT,


        bias="none",


        target_modules=[

            "q_proj",

            "k_proj",

            "v_proj",

            "o_proj"

        ],


        use_gradient_checkpointing="unsloth",


        random_state=SEED,


        use_rslora=False

    )



    model.print_trainable_parameters()



    dataset = build_dataset(

        [

            STAGE1_DATA,

            STAGE2_DATA

        ]

    )


    dataset_size = len(dataset)


    print(
        f"Training samples: {dataset_size}"
    )


    def formatting_func(batch):

        texts = []


        for messages in batch["messages"]:


            text = tokenizer.apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=False

            )


            texts.append(text)



        return {

            "text": texts

        }



    dataset = dataset.map(

        formatting_func,

        batched=True

    )

    total_tokens = 0


    for text in dataset["text"]:


        total_tokens += len(

            tokenizer(

                text,

                truncation=True,

                max_length=MAX_SEQ_LENGTH,

                add_special_tokens=False

            )["input_ids"]

        )



    print(

        f"Training tokens: {total_tokens:,}"

    )


    training_args = TrainingArguments(


        output_dir=OUTPUT_DIR,


        per_device_train_batch_size=
            PER_DEVICE_BATCH_SIZE,


        gradient_accumulation_steps=
            GRAD_ACCUMULATION,


        learning_rate=
            LEARNING_RATE,


        max_steps=
            MAX_STEPS,


        lr_scheduler_type="linear",


        optim="adamw_8bit",


        weight_decay=0.01,


        max_grad_norm=1.0,



        bf16=True,


        fp16=False,



        logging_steps=10,


        save_steps=100,


        save_total_limit=2,



        report_to="none",



        seed=SEED

    )




    trainer = SFTTrainer(


        model=model,


        tokenizer=tokenizer,


        train_dataset=dataset,


        dataset_text_field="text",


        max_seq_length=MAX_SEQ_LENGTH,


        packing=False,


        args=training_args

    )



    if torch.cuda.is_available():


        torch.cuda.reset_peak_memory_stats()



    start_time = time.time()



    trainer.train()



    end_time = time.time()

    model.save_pretrained(

        OUTPUT_DIR

    )


    tokenizer.save_pretrained(

        OUTPUT_DIR

    )



    peak_memory = None


    if torch.cuda.is_available():


        peak_memory = (

            torch.cuda.max_memory_allocated()

            /

            (1024 ** 3)

        )



    training_time = (

        end_time

        -

        start_time

    )



    report = {


        "model": {

            "name": "Qwen3-30B-A3B"

        },


        "framework": {


            "training_framework": "Unsloth",

            "trainer": "TRL SFTTrainer"


        },


        "method": {


            "name": "QLoRA",

            "trainable_component":
                "LoRA adapters only"


        },


        "software": {


            "torch":
                torch.__version__,


            "transformers":
                transformers.__version__,


            "trl":
                trl.__version__,


            "unsloth":
                unsloth.__version__

        },



        "quantization": {


            "weight_precision":
                "4-bit",


            "compute_precision":
                "BF16",


            "double_quantization":
                False

        },



        "sequence": {


            "max_seq_length":
                MAX_SEQ_LENGTH

        },



        "dataset": {


            "stage1":
                STAGE1_DATA,


            "stage2":
                STAGE2_DATA,


            "total_samples":
                dataset_size,


            "total_tokens":
                total_tokens

        },



        "lora": {


            "target_modules": [

                "q_proj",

                "k_proj",

                "v_proj",

                "o_proj"

            ],


            "rank":
                LORA_R,


            "alpha":
                LORA_ALPHA,


            "dropout":
                LORA_DROPOUT

        },



        "training": {


            "learning_rate":
                LEARNING_RATE,


            "per_device_batch_size":
                PER_DEVICE_BATCH_SIZE,


            "gradient_accumulation_steps":
                GRAD_ACCUMULATION,


            "effective_batch_size":

                (
                    PER_DEVICE_BATCH_SIZE

                    *

                    GRAD_ACCUMULATION
                ),


            "max_steps":
                MAX_STEPS,


            "estimated_epochs":

                (
                    MAX_STEPS

                    *

                    PER_DEVICE_BATCH_SIZE

                    *

                    GRAD_ACCUMULATION

                    /

                    dataset_size
                ),



            "optimizer":
                "adamw_8bit",


            "scheduler":
                "linear",


            "weight_decay":
                0.01,


            "gradient_clip":
                1.0

        },



        "resources": {


            "peak_memory_GB":
                peak_memory,


            "training_time_seconds":
                training_time

        }


    }


    report_path = os.path.join(

        OUTPUT_DIR,

        "experiment_report.json"

    )


    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            report,

            f,

            ensure_ascii=False,

            indent=2

        )



    print(

        json.dumps(

            report,

            ensure_ascii=False,

            indent=2

        )

    )


if __name__ == "__main__":


    main()