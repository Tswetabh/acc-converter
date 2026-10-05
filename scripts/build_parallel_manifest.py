"""Build parallel training, validation, and test manifests for L2-ARCTIC and CMU ARCTIC.

Pairs:
  - Male Source (ASI, RRBI) <-> Male Target (rms)
  - Female Source (SVBI, TNI) <-> Female Target (slt)
  - Unseen Target Evaluation: Male Source (ASI) <-> Unseen Male Target (bdl)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path


def extract_prompt_id(filename: str) -> str | None:
    """Extract standard prompt ID like 'arctic_a0001' or 'arctic_b0539'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(arctic_[ab]\d{4})", base, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def index_wav_files(dir_path: str) -> dict[str, str]:
    """Find all .wav files in dir_path and map prompt_id -> absolute_path."""
    files = glob.glob(os.path.join(dir_path, "**", "*.wav"), recursive=True)
    mapping = {}
    for f in files:
        pid = extract_prompt_id(f)
        if pid:
            mapping[pid] = os.path.abspath(f)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Build parallel dataset manifests.")
    parser.add_argument(
        "--l2-dir",
        default="datasets/l2arctic_release_v5.0",
        help="Path to L2-ARCTIC directory containing ASI, RRBI, SVBI, TNI",
    )
    parser.add_argument(
        "--cmu-dir",
        default="datasets/extracted/cmu_arctic",
        help="Path to extracted CMU ARCTIC directory containing cmu_us_rms_arctic, slt, bdl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/manifests",
        help="Output directory for JSON manifests",
    )
    parser.add_argument(
        "--test-cutoff",
        type=int,
        default=1000,
        help="Prompt sentence index cutoff (e.g. 1000: arctic_a0001-a0500 + b0001-b0500 in train/val, remaining in test)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Index source speakers (L2-ARCTIC Hindi)
    source_speakers = {
        "ASI": {"gender": "M", "accent": "hindi_english"},
        "RRBI": {"gender": "M", "accent": "hindi_english"},
        "SVBI": {"gender": "F", "accent": "hindi_english"},
        "TNI": {"gender": "F", "accent": "hindi_english"},
    }

    source_maps: dict[str, dict[str, str]] = {}
    for spk in source_speakers:
        spk_dir = os.path.join(args.l2_dir, spk)
        # Handle cases where folder is nested as spk/spk
        nested_spk = os.path.join(args.l2_dir, spk, spk)
        target_search = nested_spk if os.path.isdir(nested_spk) else spk_dir
        source_maps[spk] = index_wav_files(target_search)
        print(f"Indexed {len(source_maps[spk])} utterances for L2-ARCTIC speaker {spk}")

    # 2. Index target speakers (CMU ARCTIC US)
    target_speakers = ["rms", "slt", "bdl"]
    target_maps: dict[str, dict[str, str]] = {}
    for spk in target_speakers:
        spk_dir = os.path.join(args.cmu_dir, f"cmu_us_{spk}_arctic", "wav")
        target_maps[spk] = index_wav_files(spk_dir)
        print(f"Indexed {len(target_maps[spk])} utterances for CMU ARCTIC speaker {spk}")

    # 3. Create Parallel Pair Configurations
    # Source Male -> Target Male (rms)
    # Source Female -> Target Female (slt)
    pair_configs = [
        {"src_spk": "ASI", "tgt_spk": "rms", "role": "primary_male"},
        {"src_spk": "RRBI", "tgt_spk": "rms", "role": "primary_male"},
        {"src_spk": "SVBI", "tgt_spk": "slt", "role": "primary_female"},
        {"src_spk": "TNI", "tgt_spk": "slt", "role": "primary_female"},
        {"src_spk": "ASI", "tgt_spk": "bdl", "role": "eval_unseen_target"},
    ]

    all_pairs = []
    for cfg in pair_configs:
        src = cfg["src_spk"]
        tgt = cfg["tgt_spk"]
        role = cfg["role"]

        src_dict = source_maps[src]
        tgt_dict = target_maps[tgt]

        common_prompts = sorted(set(src_dict.keys()) & set(tgt_dict.keys()))
        print(f"Found {len(common_prompts)} common prompts between {src} and {tgt} ({role})")

        for pid in common_prompts:
            pair = {
                "pair_id": f"{src}_{tgt}_{pid}",
                "prompt_id": pid,
                "role": role,
                "src_speaker": src,
                "src_gender": source_speakers[src]["gender"],
                "src_accent": source_speakers[src]["accent"],
                "src_wav": src_dict[pid],
                "tgt_speaker": tgt,
                "tgt_gender": "M" if tgt in ["rms", "bdl"] else "F",
                "tgt_accent": "us_english",
                "tgt_wav": tgt_dict[pid],
            }
            all_pairs.append(pair)

    # 4. Train / Val / Test Split
    # Split by prompt_id so the same sentence doesn't leak across train and test!
    unique_prompts = sorted(list({p["prompt_id"] for p in all_pairs}))
    # Arctic prompts: 'arctic_a0001' to 'arctic_a0593', 'arctic_b0001' to 'arctic_b0539'
    # Partition ~85% train, 5% val, 10% test
    n_total = len(unique_prompts)
    n_test = max(1, int(n_total * 0.10))
    n_val = max(1, int(n_total * 0.05))
    n_train = n_total - n_test - n_val

    train_prompts = set(unique_prompts[:n_train])
    val_prompts = set(unique_prompts[n_train : n_train + n_val])
    test_prompts = set(unique_prompts[n_train + n_val :])

    train_pairs = [p for p in all_pairs if p["prompt_id"] in train_prompts and p["role"] != "eval_unseen_target"]
    val_pairs = [p for p in all_pairs if p["prompt_id"] in val_prompts and p["role"] != "eval_unseen_target"]
    test_pairs = [p for p in all_pairs if p["prompt_id"] in test_prompts or p["role"] == "eval_unseen_target"]

    print(f"\nDataset Splits by Sentence ID (No Leakage):")
    print(f"  Train sentences: {len(train_prompts)} | Total Train Pairs: {len(train_pairs)}")
    print(f"  Val sentences:   {len(val_prompts)} | Total Val Pairs:   {len(val_pairs)}")
    print(f"  Test sentences:  {len(test_prompts)} | Total Test Pairs:  {len(test_pairs)}")

    # Write manifests
    for split_name, split_data in [
        ("parallel_train.json", train_pairs),
        ("parallel_val.json", val_pairs),
        ("parallel_test.json", test_pairs),
        ("parallel_all.json", all_pairs),
    ]:
        out_path = os.path.join(args.output_dir, split_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved: {out_path} ({len(split_data)} records)")


if __name__ == "__main__":
    main()
