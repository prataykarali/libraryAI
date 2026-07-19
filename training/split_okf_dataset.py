#!/usr/bin/env python3
"""Split the cleaned OKF dataset into train / test pairs for SFT evaluation.

Splits are done at the *chunk* level so no text leaks between train and test:
all single-concept snippets and the full chunk-level example that come from the
same (doc_id, chunk_id) stay together in the same split.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "training_data"
SRC = DATA_DIR / "okf_training_pairs_v3.jsonl"
TRAIN = DATA_DIR / "okf_train_pairs_v3.jsonl"
TEST = DATA_DIR / "okf_test_pairs_v3.jsonl"
TEST_RATIO = 0.15
SEED = 42


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} total records from {SRC.name}")

    # Group by chunk so snippets from the same source chunk don't leak across splits.
    groups = defaultdict(list)
    for rec in records:
        key = (rec.get("doc_id", ""), rec.get("chunk_id", ""))
        groups[key].append(rec)

    keys = list(groups.keys())
    random.seed(SEED)
    random.shuffle(keys)

    n_test = max(1, int(len(keys) * TEST_RATIO))
    test_keys = set(keys[:n_test])

    train_records = []
    test_records = []
    for key, recs in groups.items():
        if key in test_keys:
            test_records.extend(recs)
        else:
            train_records.extend(recs)

    def write_jsonl(path, recs):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN, train_records)
    write_jsonl(TEST, test_records)

    print(f"Train: {len(train_records)} records -> {TRAIN}")
    print(f"Test:  {len(test_records)}  records -> {TEST}")
    print(f"Unique chunks: train={len(keys)-n_test}, test={n_test}")

    # Save a tiny metadata summary for fine-tuning scripts.
    meta = {
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_chunks": len(keys) - n_test,
        "test_chunks": n_test,
    }
    with open(DATA_DIR / "okf_dataset_split_v3.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
