#!/usr/bin/env python3
"""Validate JSONL feature caches produced by Stage 1.

Usage:
    python scripts/verify_cache.py outputs/tiny_validate/feature_cache
    python scripts/verify_cache.py outputs/smoke_test/feature_cache
    python scripts/verify_cache.py outputs/full_experiment/feature_cache
"""

import json
import sys
from pathlib import Path


def verify(cache_dir: str):
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        print(f"[FAIL] cache directory does not exist: {cache_dir}")
        return False

    all_ok = True
    for split in ("train_features.jsonl", "test_features.jsonl"):
        fpath = cache_path / split
        if not fpath.exists():
            print(f"  [WARN] missing file: {split}")
            all_ok = False
            continue

        lines = []
        errors = 0
        with open(fpath) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [FAIL] invalid JSON at line {i}: {e}")
                    errors += 1

        if errors == 0 and lines:
            # Check first and last samples.
            first = json.loads(lines[0])
            last = json.loads(lines[-1])
            required_keys = {
                "id", "label", "lang",
                "module1_text_emb", "module1_video_emb", "module1_audio_emb",
                "module2_branches", "module2_mask",
                "module3_bert_cls", "text_aux",
            }
            first_keys = set(first.keys())
            missing = required_keys - first_keys
            if missing:
                print(f"  [FAIL] {split}: missing keys: {missing}")
                all_ok = False
            else:
                print(f"  [PASS] {split}: {len(lines)} rows, "
                      f"first={first['id']}({first['lang']}) last={last['id']}({last['lang']})")
        elif errors > 0:
            print(f"  [FAIL] {split}: {errors} invalid rows out of {len(lines)}")
            all_ok = False
        else:
            print(f"  [WARN] {split}: empty file")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_cache.py <cache_dir>")
        sys.exit(1)

    ok = verify(sys.argv[1])
    print(f"\nResult: {'cache is valid' if ok else 'cache has errors'}")
    sys.exit(0 if ok else 1)
