# IFC-MoE-FineTuning

This repository provides the datasets, QLoRA adapter weights, and evaluation scripts associated with the study:

**Integrating semantic and operational capabilities in IFC-compliant BIM for the building design phase: A mixture-of-experts fine-tuning approach**

## Repository structure

```text
IFC-MoE-FineTuning/
├── code/
│   ├── train_qwen3_ifc_qlora.py
│   ├── generate_base_semantic.py
│   ├── generate_finetuned_semantic.py
│   ├── evaluate_base_semantic_metrics.py
│   ├── evaluate_finetuned_semantic_metrics.py
│   ├── measure_base_cold_run.py
│   └── measure_finetuned_cold_run.py
├── datasets/
│   ├── ifc4_semantic_train.json
│   ├── ifc4_operational_train.json
│   └── ifc4_test.json
└── qlora_weights/
    └── qwen3_ifc_qlora/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

`code/` contains the scripts for QLoRA fine-tuning, semantic response generation, semantic evaluation, and cold-run measurement.

`datasets/` contains the IFC semantic and operational datasets used for training and testing.

`qlora_weights/` contains the trained QLoRA adapter for Qwen3-30B-A3B.

## Dataset split

The semantic dataset is constructed from the IFC4 ADD2 TC1 EXPRESS schema and the corresponding official concept templates. The top 600 semantic items are split at the semantic-item level into 480 training items and 120 test items, producing 4,306 training QA pairs and 1,748 test QA pairs.

The operational dataset contains query, edit, and create tasks. It includes 986 training samples and 216 test samples.

| Dataset | Training | Test |
|---|---:|---:|
| Semantic | 4,306 | 1,748 |
| Operational | 986 | 216 |
| Total | 5,292 | 1,964 |

The semantic training data are stored in `datasets/ifc4_semantic_train.json`, the operational training data in `datasets/ifc4_operational_train.json`, and the test data in `datasets/ifc4_test.json`.

## Experimental environment

The experiments reported in the manuscript were conducted using:

```text
CPU: Intel Xeon Platinum 8470Q
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
GPU memory: 94.97 GB

Python: 3.12.12
PyTorch: 2.10.0+cu128
CUDA: 12.8
cuDNN: 91002
Transformers: 4.57.6
Training framework: Unsloth
```

The base model is **Qwen3-30B-A3B**. The original base model is not included in this repository and should be obtained separately from its official source.

## Main commands

Before running the scripts, update the local model, dataset, and output paths where necessary.

### Fine-tuning

```bash
python code/train_qwen3_ifc_qlora.py
```

### Generate semantic responses

```bash
python code/generate_base_semantic.py
python code/generate_finetuned_semantic.py
```

### Evaluate semantic performance

```bash
python code/evaluate_base_semantic_metrics.py
python code/evaluate_finetuned_semantic_metrics.py
```

### Measure semantic cold-run performance

```bash
python code/measure_base_cold_run.py
python code/measure_finetuned_cold_run.py
```

## Notes

The QLoRA adapter is provided in:

```text
qlora_weights/qwen3_ifc_qlora/
```

The operational dataset contains task definitions and validated operation trajectories. Full operational execution additionally depends on the IFC models and the Blender/Bonsai MCP environment used in the study.

Latency and memory measurements may vary across hardware and software environments.

## Data availability

The data, code, and QLoRA adapter weights supporting this study are provided in this repository.
