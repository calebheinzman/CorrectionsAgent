from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Tuple

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from mock_apis.cloud_apis import mock_cloud_watch, mock_dynamodb, mock_model_registry, mock_s3


Label = Literal["relevant", "irrelevant"]


@dataclass(frozen=True)
class Example:
    text: str
    label: Label


def _utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def _read_jsonl(path: Path) -> List[Example]:
    rows: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(Example(text=str(obj["text"]), label=obj["label"]))
    return rows


class NaiveBayesTextClassifier:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.class_log_prior: Dict[str, float] = {}
        self.token_log_prob: Dict[str, Dict[str, float]] = {}
        self.vocab: set[str] = set()
        self.unk_log_prob: Dict[str, float] = {}

    def fit(self, X: List[str], y: List[str]) -> None:
        docs_by_label: Dict[str, List[List[str]]] = defaultdict(list)
        for text, label in zip(X, y):
            docs_by_label[label].append(_tokenize(text))

        n_docs = len(X)
        for label in self.labels:
            self.class_log_prior[label] = math.log(
                (len(docs_by_label[label]) + 1) / (n_docs + len(self.labels))
            )

        token_counts_by_label: Dict[str, Counter[str]] = {}
        total_tokens_by_label: Dict[str, int] = {}

        for label in self.labels:
            c = Counter()
            for toks in docs_by_label[label]:
                c.update(toks)
            token_counts_by_label[label] = c
            total_tokens_by_label[label] = sum(c.values())
            self.vocab.update(c.keys())

        vocab_size = max(len(self.vocab), 1)

        for label in self.labels:
            denom = total_tokens_by_label[label] + vocab_size
            self.token_log_prob[label] = {
                tok: math.log((token_counts_by_label[label].get(tok, 0) + 1) / denom)
                for tok in self.vocab
            }
            self.unk_log_prob[label] = math.log(1 / denom)

    def predict(self, X: List[str]) -> List[str]:
        preds: List[str] = []
        for text in X:
            toks = _tokenize(text)
            best_label = None
            best_score = None
            for label in self.labels:
                score = self.class_log_prior[label]
                tlp = self.token_log_prob[label]
                unk = self.unk_log_prob[label]
                for tok in toks:
                    score += tlp.get(tok, unk)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = label
            preds.append(str(best_label))
        return preds


def _split_train_test(rows: List[Example], seed: int, test_ratio: float) -> Tuple[List[Example], List[Example]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n_test = max(1, int(len(rows) * test_ratio))
    return rows[n_test:], rows[:n_test]


def _accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / max(len(y_true), 1)


def _confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    m: Dict[str, Dict[str, int]] = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(y_true, y_pred):
        if a in m and b in m[a]:
            m[a][b] += 1
    return m


def _resolve_data_path(raw: str) -> Path:
    data_path = Path(raw)
    if data_path.exists():
        return data_path

    candidate_paths = [
        Path(__file__).resolve().parent / "data" / data_path.name,
        Path(__file__).resolve().parent / data_path.name,
    ]
    for p in candidate_paths:
        if p.exists():
            return p

    attempted = [str(data_path)] + [str(p) for p in candidate_paths]
    raise FileNotFoundError(
        "Could not find --data file. Tried:\n" + "\n".join(f"- {p}" for p in attempted)
    )


def _latest_data_file(prefix: str) -> Path:
    data_dir = Path(__file__).resolve().parent / "data"
    candidates = list(data_dir.glob(f"{prefix}*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No training data files found in {data_dir} matching {prefix}*.jsonl"
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, help="Path to relevance_check_*.jsonl")
    parser.add_argument(
        "--models-dir",
        default=str(Path(__file__).resolve().parent / "models"),
        help="Directory for local artifacts",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--s3-bucket", default=os.getenv("TRAINING_S3_BUCKET", "training"))
    args = parser.parse_args()

    if args.data:
        data_path = _resolve_data_path(args.data)
    else:
        data_path = _latest_data_file(prefix="relevance_check_")
    rows = _read_jsonl(data_path)
    train_rows, test_rows = _split_train_test(rows, seed=args.seed, test_ratio=args.test_ratio)

    X_train = [r.text for r in train_rows]
    y_train = [r.label for r in train_rows]
    X_test = [r.text for r in test_rows]
    y_test = [r.label for r in test_rows]

    clf = NaiveBayesTextClassifier(labels=["relevant", "irrelevant"])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = _accuracy(y_test, y_pred)
    conf = _confusion(y_test, y_pred, labels=["relevant", "irrelevant"])

    run_id = f"relevance_check_{_utc_compact_ts()}"

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = models_dir / f"{run_id}.pkl"
    local_metrics_path = models_dir / f"{run_id}.metrics.json"

    artifact = {
        "task": "relevance_check",
        "run_id": run_id,
        "model_type": "naive_bayes_bow",
        "labels": ["relevant", "irrelevant"],
        "model": clf,
    }

    local_model_path.write_bytes(pickle.dumps(artifact))

    metrics = {
        "task": "relevance_check",
        "run_id": run_id,
        "data_path": str(data_path),
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "accuracy": acc,
        "confusion": conf,
    }
    local_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    s3_model_key = f"models/{run_id}.pkl"
    s3_metrics_key = f"metrics/{run_id}.json"
    mock_s3.put_object(args.s3_bucket, s3_model_key, local_model_path.read_bytes())
    mock_s3.put_object(
        args.s3_bucket, s3_metrics_key, local_metrics_path.read_bytes()
    )

    mock_dynamodb.put_item(
        "training_runs",
        {
            "pk": f"training_run#{run_id}",
            "run_id": run_id,
            "task": "relevance_check",
            "timestamp": _utc_compact_ts(),
            "s3_bucket": args.s3_bucket,
            "s3_model_key": s3_model_key,
            "s3_metrics_key": s3_metrics_key,
            "accuracy": acc,
        },
    )

    mock_cloud_watch.put_metric(
        "relevance_check.accuracy",
        float(acc),
        dimensions={"run_id": run_id, "task": "relevance_check"},
    )

    mock_model_registry.set_current_model(
        provider="local_nb",
        model_id=run_id,
        version="1",
        task="relevance_check",
    )

    print(
        "\n".join(
            [
                f"trained: {run_id}",
                f"task: relevance_check",
                f"accuracy: {acc:.4f}",
                f"local_model: {local_model_path}",
                f"local_metrics: {local_metrics_path}",
                f"mock_s3: s3://{args.s3_bucket}/{s3_model_key}",
                f"mock_s3: s3://{args.s3_bucket}/{s3_metrics_key}",
            ]
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
