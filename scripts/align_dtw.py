"""Dynamic Time Warping (DTW) feature alignment script.

Processes parallel pairs from data/manifests/parallel_train.json,
extracts/caches HuBERT representations, aligns them frame-by-frame,
and saves the aligned parallel pairs ready for Conformer training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from accent_converter.audio.resampler import resample
from accent_converter.models.backends.accent.dtw import (
    align_hubert_with_path,
    source_length_target,
)
from accent_converter.models.backends.content.hubert import HubertContentEncoder


def get_cached_or_compute_hubert(
    wav_path: str,
    spk_id: str,
    prompt_id: str,
    cache_dir: str,
    hubert_encoder: HubertContentEncoder,
) -> torch.Tensor:
    """Get HuBERT representation from cache or extract and cache it."""
    cache_file = os.path.join(cache_dir, f"{spk_id}_{prompt_id}.pt")
    if os.path.isfile(cache_file):
        try:
            return torch.load(cache_file, weights_only=True)
        except Exception:
            pass

    # Read and resample if necessary
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample(audio.astype(np.float32), sr, 16000)
    else:
        audio = audio.astype(np.float32)

    # Encode with HuBERT
    feat_np = hubert_encoder.encode(audio)
    feat_tensor = torch.from_numpy(feat_np).float()

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(feat_tensor, cache_file)
    return feat_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract & align HuBERT features using DTW.")
    parser.add_argument(
        "--manifest",
        default="data/manifests/parallel_train.json",
        help="Path to parallel pairs JSON manifest",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/features/hubert",
        help="Directory to cache extracted HuBERT features",
    )
    parser.add_argument(
        "--output-dir",
        default="data/aligned/arctic_train",
        help="Directory to save aligned tensor pairs",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for HuBERT encoder",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of pairs to process (useful for testing)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.manifest):
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Loaded {len(pairs)} pairs from {args.manifest}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("Loading HuBERT Content Encoder...")
    hubert = HubertContentEncoder()
    hubert.load(device=args.device)

    start_time = time.time()
    processed = 0
    skipped = 0

    pbar = tqdm(pairs, desc="Aligning Pairs")
    for pair in pbar:
        pair_id = pair["pair_id"]
        out_path = os.path.join(args.output_dir, f"{pair_id}.pt")

        if os.path.isfile(out_path):
            skipped += 1
            continue

        src_wav = pair["src_wav"]
        tgt_wav = pair["tgt_wav"]
        src_spk = pair["src_speaker"]
        tgt_spk = pair["tgt_speaker"]
        prompt_id = pair["prompt_id"]

        try:
            h_src = get_cached_or_compute_hubert(src_wav, src_spk, prompt_id, args.cache_dir, hubert)
            h_tgt = get_cached_or_compute_hubert(tgt_wav, tgt_spk, prompt_id, args.cache_dir, hubert)

            path_x, path_y, cost = align_hubert_with_path(h_src, h_tgt, use_fastdtw=True)
            px = torch.from_numpy(path_x)
            py = torch.from_numpy(path_y)
            tgt_srclen = source_length_target(h_tgt, path_x, path_y, h_src.shape[0])

            record = {
                "pair_id": pair_id,
                "prompt_id": prompt_id,
                "src_speaker": src_spk,
                "tgt_speaker": tgt_spk,
                "src_gender": pair["src_gender"],
                "tgt_gender": pair["tgt_gender"],
                # frame-repeated aligned sequences (length = path length)
                "src_hubert": h_src[px].cpu(),
                "tgt_hubert": h_tgt[py].cpu(),
                # source-length view: no repeated frames (preferred for training)
                "src_orig": h_src.cpu(),
                "tgt_srclen": tgt_srclen.cpu(),
                "path_src": px,
                "path_tgt": py,
                "src_len": int(h_src.shape[0]),
                "tgt_len": int(h_tgt.shape[0]),
                "dtw_cost": cost,
                "aligned_len": int(len(path_x)),
            }
            torch.save(record, out_path)
            processed += 1
        except Exception as e:
            print(f"\nWarning: Failed to align {pair_id}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\nDTW Alignment Complete!")
    print(f"  Processed: {processed} pairs")
    print(f"  Skipped (already done): {skipped} pairs")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed / max(1, processed):.2f}s/pair)")
    print(f"  Saved in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
