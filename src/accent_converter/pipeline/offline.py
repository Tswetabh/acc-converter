"""
pipeline.offline
================
Full-utterance offline accent conversion pipeline.

This module implements the end-to-end offline pipeline:
1. Audio loading & format enforcement (soundfile, resampler, normalizer).
2. Noise suppression (spectral gating via NoiseSuppressor).
3. Speaker identity extraction from enrollment audio (SpeakerEncoder -> (192,)).
4. Linguistic content representation extraction (ContentEncoder -> (T, 768)).
5. Accent translation (AccentTranslator -> (T, 768)).
6. Waveform synthesis (SpeechSynthesizer -> 16 kHz float32 waveform).
7. Optional writing to disk.

Strict architectural principles enforced:
- Only abstract interfaces and factory constructors are used.
- Speaker identity is conditioned strictly at the synthesizer, not the translator.
- Audio format is canonically 16 kHz mono float32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf

from accent_converter.core.config import load_config
from accent_converter.audio.normalizer import normalize, CANONICAL_SAMPLE_RATE
from accent_converter.audio.resampler import resample
from accent_converter.audio.preprocessing.interfaces import NoiseSuppressor
from accent_converter.audio.preprocessing.factory import build_noise_suppressor
from accent_converter.models.interfaces import (
    ContentEncoder,
    SpeakerEncoder,
    AccentTranslator,
    SpeechSynthesizer,
)
from accent_converter.models.factory import (
    build_content_encoder,
    build_speaker_encoder,
    build_accent_translator,
    build_synthesizer,
)

VALID_ACCENTS: tuple[str, ...] = ("us", "uk", "neutral")


def run_offline_pipeline(
    audio_path: str | Path,
    target_accent: str,
    enrollment_audio_path: str | Path,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    device: str | None = None,
    *,
    content_encoder: ContentEncoder | None = None,
    speaker_encoder: SpeakerEncoder | None = None,
    accent_translator: AccentTranslator | None = None,
    synthesizer: SpeechSynthesizer | None = None,
    noise_suppressor: NoiseSuppressor | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Full-utterance offline accent conversion pipeline: WAV -> WAV.

    Parameters
    ----------
    audio_path:
        Path to source audio file to convert.
    target_accent:
        Target accent code ('us', 'uk', 'neutral').
    enrollment_audio_path:
        Path to enrollment/reference audio for speaker identity extraction.
    config_path:
        Path to YAML config. If None and config is None, default_config.yaml is used.
    output_path:
        Optional path to write converted 16 kHz WAV audio.
    device:
        Optional device override ('cpu', 'cuda').
    content_encoder:
        Optional pre-instantiated ContentEncoder (for dependency injection / testing).
    speaker_encoder:
        Optional pre-instantiated SpeakerEncoder (for dependency injection / testing).
    accent_translator:
        Optional pre-instantiated AccentTranslator (for dependency injection / testing).
    synthesizer:
        Optional pre-instantiated SpeechSynthesizer (for dependency injection / testing).
    noise_suppressor:
        Optional pre-instantiated NoiseSuppressor (for dependency injection / testing).
    config:
        Optional pre-loaded configuration dictionary.

    Returns
    -------
    np.ndarray
        1-D float32 converted audio waveform at 16 kHz.
    """
    # 1. Target accent validation
    norm_accent = str(target_accent).strip().lower()
    if norm_accent not in VALID_ACCENTS:
        raise ValueError(
            f"Invalid target_accent: '{target_accent}'. Expected one of {VALID_ACCENTS}"
        )

    # 2. Path validation
    src_path = Path(audio_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source audio file not found: {audio_path}")

    ref_path = Path(enrollment_audio_path)
    if not ref_path.is_file():
        raise FileNotFoundError(f"Enrollment audio file not found: {enrollment_audio_path}")

    # 3. Load configuration if not provided
    if config is None:
        if config_path is None:
            resolved_cfg_path = Path(__file__).resolve().parents[3] / "configs" / "default_config.yaml"
        else:
            resolved_cfg_path = Path(config_path)

        if not resolved_cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_cfg_path}")
        config = load_config(str(resolved_cfg_path))

    # 4. Load & preprocess source audio
    raw_source, sr_source = sf.read(str(src_path), dtype="float32")
    if len(raw_source) == 0:
        raise ValueError("Source audio file is empty.")
    if sr_source != CANONICAL_SAMPLE_RATE:
        raw_source = resample(raw_source, orig_sr=sr_source, target_sr=CANONICAL_SAMPLE_RATE)
    source_audio = normalize(raw_source)

    # 5. Load & preprocess enrollment audio
    raw_ref, sr_ref = sf.read(str(ref_path), dtype="float32")
    if len(raw_ref) == 0:
        raise ValueError("Enrollment audio file is empty.")
    if sr_ref != CANONICAL_SAMPLE_RATE:
        raw_ref = resample(raw_ref, orig_sr=sr_ref, target_sr=CANONICAL_SAMPLE_RATE)
    enrollment_audio = normalize(raw_ref)

    # 6. Noise suppression
    if noise_suppressor is None:
        ns_cfg = config.get("preprocessing", {}).get("noise_suppressor", {"backend": "noisereduce"})
        noise_suppressor = build_noise_suppressor(ns_cfg)
    cleaned_audio = noise_suppressor.process(source_audio)

    # 7. Speaker identity embedding extraction
    if speaker_encoder is None:
        spk_cfg = config.get("models", {}).get("speaker_encoder", {"backend": "ecapa"})
        speaker_encoder = build_speaker_encoder(spk_cfg)
        speaker_encoder.load(checkpoint_path=spk_cfg.get("checkpoint"), device=device)
    speaker_embedding = speaker_encoder.encode(enrollment_audio)

    # 8. Linguistic content representation extraction
    if content_encoder is None:
        cnt_cfg = config.get("models", {}).get("content_encoder", {"backend": "hubert"})
        content_encoder = build_content_encoder(cnt_cfg)
        content_encoder.load(checkpoint_path=cnt_cfg.get("checkpoint"), device=device)
    content_rep = content_encoder.encode(cleaned_audio)

    # 9. Accent translation
    if accent_translator is None:
        trans_cfg = config.get("models", {}).get("accent_translator", {"backend": "naive"})
        accent_translator = build_accent_translator(trans_cfg)
        accent_translator.load(checkpoint_path=trans_cfg.get("checkpoint"))
    target_rep = accent_translator.translate(content_rep, target_accent=norm_accent)

    # 10. Waveform synthesis
    if synthesizer is None:
        synth_cfg = config.get("synthesis", {"backend": "vocoder"})
        synthesizer = build_synthesizer(synth_cfg)
        synthesizer.load(checkpoint_path=synth_cfg.get("checkpoint"))
    converted_waveform = synthesizer.synthesize(target_rep, speaker_embedding=speaker_embedding)

    # 11. Write output if requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), converted_waveform, CANONICAL_SAMPLE_RATE)

    return converted_waveform
