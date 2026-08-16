import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import random
import numpy as np
import torch

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


SEED = 3407
BOOTSTRAP_N = 1000

INPUT_JSON = "../outputs/base_generated_responses.json"
OUTPUT_JSON = "../outputs/base_semantic_evaluation_metrics.json"

BERT_SCORE_MODEL_PATH = "FacebookAI/roberta-large"
SBERT_MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"

DEVICE = "cuda"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def bootstrap_ci(values, n=BOOTSTRAP_N, ci=0.95):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(values, size=(n, len(values)), replace=True)
    means = samples.mean(axis=1)

    return {
        "mean": float(values.mean()),
        "95%_CI": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))
        ]
    }


def compute_rouge(generated, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    r1, r2, rl = [], [], []

    for g, r in zip(generated, references):
        score = scorer.score(r, g)
        r1.append(score["rouge1"].fmeasure)
        r2.append(score["rouge2"].fmeasure)
        rl.append(score["rougeL"].fmeasure)

    return r1, r2, rl


def compute_bert(generated, references):
    P, R, F1 = bert_score_fn(
        generated,
        references,
        model_type=BERT_SCORE_MODEL_PATH,
        num_layers=17,
        rescale_with_baseline=False,
        device=DEVICE,
        batch_size=16
    )

    return P.tolist(), R.tolist(), F1.tolist()


print("Loading SBERT model...")
tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL_PATH)
model = AutoModel.from_pretrained(SBERT_MODEL_PATH).to(DEVICE).eval()


def mean_pooling(output, mask):
    token_embeddings = output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode(texts, batch_size=32):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output = model(**inputs)

        emb = mean_pooling(
            output,
            inputs["attention_mask"]
        )

        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb)

    return torch.cat(embeddings, dim=0)


def compute_sbert(generated, references):
    gen_emb = encode(generated)
    ref_emb = encode(references)

    scores = (gen_emb * ref_emb).sum(dim=1)

    return scores.cpu().tolist()


def main():

    print("Loading generated responses...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    generated = [
        x["generated"].strip()
        for x in data
    ]

    references = [
        x["reference"].strip()
        for x in data
    ]

    print("Samples:", len(generated))

    print("Computing ROUGE...")
    rouge1, rouge2, rougeL = compute_rouge(
        generated,
        references
    )

    print("Computing BERTScore...")
    bert_p, bert_r, bert_f1 = compute_bert(
        generated,
        references
    )

    print("Computing SBERT...")
    sbert = compute_sbert(
        generated,
        references
    )


    results = {

        "input_file": INPUT_JSON,

        "number_of_samples": len(generated),

        "metrics": {

            "ROUGE-1 (95% CI)": bootstrap_ci(rouge1),

            "ROUGE-2 (95% CI)": bootstrap_ci(rouge2),

            "ROUGE-L (95% CI)": bootstrap_ci(rougeL),

            "BERTScore Precision (95% CI)": bootstrap_ci(bert_p),

            "BERTScore Recall (95% CI)": bootstrap_ci(bert_r),

            "BERTScore F1 (95% CI)": bootstrap_ci(bert_f1),

            "SBERT Score (95% CI)": bootstrap_ci(sbert)

        }
    }


    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4
        )


    print("\nEvaluation completed.")
    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main()