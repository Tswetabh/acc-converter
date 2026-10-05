"""
scripts.download_pretrained
===========================
Utility to download and cache pretrained neural models needed for the Accent Converter:
1. SpeechBrain 16 kHz HiFi-GAN Vocoder (``speechbrain/tts-hifigan-libritts-16kHz``)
2. HuggingFace HuBERT Base Content Encoder (``facebook/hubert-base-ls960``)
3. SpeechBrain ECAPA-TDNN Speaker Encoder (``speechbrain/spkrec-ecapa-voxceleb``)

Usage
-----
    python scripts/download_pretrained.py --model hifigan
    python scripts/download_pretrained.py --model all
"""

from __future__ import annotations

import argparse
import os
import sys


def download_hifigan(destination_dir: str = "models/hifigan_16k", device: str = "cpu") -> None:
    print(f"[1/3] Downloading/Verifying HiFi-GAN 16 kHz (speechbrain/tts-hifigan-libritts-16kHz)...")
    os.makedirs(destination_dir, exist_ok=True)
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.utils.fetching import LocalStrategy

    strategy = LocalStrategy.COPY if os.name == "nt" else LocalStrategy.SYMLINK
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir=destination_dir,
        run_opts={"device": device},
        local_strategy=strategy,
    )
    print(f"      HiFi-GAN ready at {destination_dir}")


def download_hubert(cache_dir: str | None = None) -> None:
    print(f"[2/3] Downloading/Verifying HuBERT Base (facebook/hubert-base-ls960)...")
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960", cache_dir=cache_dir)
    HubertModel.from_pretrained("facebook/hubert-base-ls960", cache_dir=cache_dir)
    print("      HuBERT Base ready.")


def download_ecapa(destination_dir: str = "models/ecapa", device: str = "cpu") -> None:
    print(f"[3/3] Downloading/Verifying ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
    os.makedirs(destination_dir, exist_ok=True)
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    strategy = LocalStrategy.COPY if os.name == "nt" else LocalStrategy.SYMLINK
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=destination_dir,
        run_opts={"device": device},
        local_strategy=strategy,
    )
    print(f"      ECAPA-TDNN ready at {destination_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pretrained checkpoints for Accent Converter.")
    parser.add_argument(
        "--model",
        choices=["hifigan", "hubert", "ecapa", "all"],
        default="all",
        help="Which model(s) to download (default: all)",
    )
    parser.add_argument(
        "--hifigan-dir",
        default="models/hifigan_16k",
        help="Directory to save HiFi-GAN checkpoint",
    )
    parser.add_argument(
        "--ecapa-dir",
        default="models/ecapa",
        help="Directory to save ECAPA checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for initialization",
    )
    args = parser.parse_args()

    try:
        if args.model in ("hifigan", "all"):
            download_hifigan(args.hifigan_dir, args.device)
        if args.model in ("hubert", "all"):
            download_hubert()
        if args.model in ("ecapa", "all"):
            download_ecapa(args.ecapa_dir, args.device)
        print("\nAll requested pretrained checkpoints are ready!")
        return 0
    except Exception as exc:
        print(f"\nERROR: Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
