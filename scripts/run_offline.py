"""
scripts/run_offline.py
======================
CLI entry point for running the full-utterance offline conversion pipeline.

Usage
-----
::

    python scripts/run_offline.py input.wav --accent us --enrollment ref.wav --output converted.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from accent_converter.pipeline.offline import run_offline_pipeline, VALID_ACCENTS


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline accent conversion pipeline on a WAV audio file."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to source audio file to convert.",
    )
    parser.add_argument(
        "--accent",
        "-a",
        type=str,
        default="us",
        choices=VALID_ACCENTS,
        help="Target accent (us, uk, neutral). Default: us",
    )
    parser.add_argument(
        "--enrollment",
        "-e",
        type=str,
        required=True,
        help="Path to speaker enrollment/reference audio file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output converted WAV file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional path to config YAML.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Device override ('cpu', 'cuda').",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Accent Converter — Offline Pipeline")
    print(f"Source audio:      {args.input}")
    print(f"Enrollment audio:  {args.enrollment}")
    print(f"Target accent:     {args.accent}")
    print(f"Output path:       {args.output or '(none)'}")
    print("=" * 60)

    t0 = time.perf_counter()
    try:
        waveform = run_offline_pipeline(
            audio_path=args.input,
            target_accent=args.accent,
            enrollment_audio_path=args.enrollment,
            config_path=args.config,
            output_path=args.output,
            device=args.device,
        )
    except Exception as exc:
        print(f"Error during offline pipeline execution: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    duration_s = len(waveform) / 16000.0
    print(f"Conversion complete in {elapsed:.3f} s.")
    print(f"Output samples: {len(waveform)} ({duration_s:.2f} s @ 16 kHz)")
    if args.output:
        print(f"Saved converted audio to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
