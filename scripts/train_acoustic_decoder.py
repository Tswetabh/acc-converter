"""
scripts.train_acoustic_decoder
==============================
Training pipeline for the Neural Acoustic Decoder:
Maps (HuBERT features + ECAPA speaker embedding) -> 80-bin Mel spectrogram.

Features
--------
1. Offline Feature Caching: Extracts HuBERT, ECAPA, and Mel spectrograms once
   and caches them to disk as .pt tensors for fast GPU training.
2. Variable Length Batching: Custom collate_fn handles variable utterance lengths
   with masked L1 reconstruction loss.
3. Checkpoint Management: Saves best and latest model weights, compatible with
   VocoderSynthesizer.load().

Usage
-----
    # Extract features and train on CMU ARCTIC / speech WAV files
    python scripts/train_acoustic_decoder.py --data-dir data/raw/cmu_arctic --epochs 50 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accent_converter.audio.mel import extract_mel_spectrogram
from accent_converter.models.backends.content.hubert import HubertContentEncoder
from accent_converter.models.backends.speaker.ecapa import EcapaSpeakerEncoder
from accent_converter.models.backends.synthesis.acoustic_decoder import AcousticDecoder


class SpeechMelDataset(Dataset):
    """Dataset loading pre-extracted (HuBERT, ECAPA, Mel) feature triplets."""

    def __init__(self, file_triplets: list[tuple[str, str, str]]) -> None:
        self.triplets = file_triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hubert_path, ecapa_path, mel_path = self.triplets[idx]
        hubert = torch.load(hubert_path, map_location="cpu")
        ecapa = torch.load(ecapa_path, map_location="cpu")
        mel = torch.load(mel_path, map_location="cpu")

        if isinstance(hubert, np.ndarray):
            hubert = torch.from_numpy(hubert)
        if isinstance(ecapa, np.ndarray):
            ecapa = torch.from_numpy(ecapa)
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)

        return {
            "hubert": hubert.float(),    # (T_hubert, 768)
            "ecapa": ecapa.float(),      # (192,)
            "mel": mel.float(),          # (80, T_mel)
        }


def collate_speech_mel(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """Collate variable length feature sequences with padding."""
    hubert_list = [item["hubert"] for item in batch]
    ecapa_list = [item["ecapa"] for item in batch]
    mel_list = [item["mel"] for item in batch]

    hubert_lens = torch.tensor([h.shape[0] for h in hubert_list], dtype=torch.long)
    mel_lens = torch.tensor([m.shape[1] for m in mel_list], dtype=torch.long)

    max_hubert_len = max(h.shape[0] for h in hubert_list)
    max_mel_len = max(m.shape[1] for m in mel_list)

    B = len(batch)
    hubert_padded = torch.zeros((B, max_hubert_len, 768), dtype=torch.float32)
    mel_padded = torch.zeros((B, 80, max_mel_len), dtype=torch.float32)

    for i in range(B):
        h_len = hubert_list[i].shape[0]
        m_len = mel_list[i].shape[1]
        hubert_padded[i, :h_len] = hubert_list[i]
        mel_padded[i, :, :m_len] = mel_list[i]

    ecapa_tensor = torch.stack(ecapa_list, dim=0)  # (B, 192)

    return {
        "hubert": hubert_padded,
        "ecapa": ecapa_tensor,
        "mel": mel_padded,
        "hubert_lens": hubert_lens,
        "mel_lens": mel_lens,
    }


def extract_features(
    wav_files: list[str],
    cache_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[tuple[str, str, str]]:
    """Extract and cache HuBERT, ECAPA, and Mel features for audio files."""
    hubert_dir = os.path.join(cache_dir, "hubert")
    ecapa_dir = os.path.join(cache_dir, "ecapa")
    mel_dir = os.path.join(cache_dir, "mel")
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(ecapa_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)

    print(f"Loading pretrained encoders for feature caching on {device}...")
    hubert_enc = HubertContentEncoder()
    hubert_enc.load(device=device)

    ecapa_enc = EcapaSpeakerEncoder()
    ecapa_enc.load(device=device)

    import scipy.io.wavfile as wavfile
    from accent_converter.audio.resampler import resample as resample_audio

    triplets = []
    print(f"Extracting features for {len(wav_files)} audio files...")
    start_time = time.time()

    for idx, wav_path in enumerate(wav_files):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        h_path = os.path.join(hubert_dir, f"{utt_id}.pt")
        e_path = os.path.join(ecapa_dir, f"{utt_id}.pt")
        m_path = os.path.join(mel_dir, f"{utt_id}.pt")

        if os.path.isfile(h_path) and os.path.isfile(e_path) and os.path.isfile(m_path):
            triplets.append((h_path, e_path, m_path))
            continue

        try:
            sr, raw_audio = wavfile.read(wav_path)
            if raw_audio.dtype == np.int16:
                audio_float = raw_audio.astype(np.float32) / 32768.0
            elif raw_audio.dtype == np.int32:
                audio_float = raw_audio.astype(np.float32) / 2147483648.0
            else:
                audio_float = raw_audio.astype(np.float32)

            if audio_float.ndim > 1:
                audio_float = audio_float.mean(axis=1)

            if sr != 16000:
                audio_float = resample_audio(audio_float, sr, 16000)

            # Skip very short audio (< 0.2s)
            if len(audio_float) < 3200:
                continue

            # 1. HuBERT content representation
            hubert_rep = hubert_enc.encode(audio_float)  # (T, 768)

            # 2. ECAPA speaker embedding
            ecapa_emb = ecapa_enc.encode(audio_float)    # (192,)

            # 3. Mel spectrogram
            mel_spec = extract_mel_spectrogram(audio_float, sample_rate=16000)  # (80, T_mel)

            torch.save(torch.from_numpy(hubert_rep), h_path)
            torch.save(torch.from_numpy(ecapa_emb), e_path)
            torch.save(mel_spec.cpu(), m_path)

            triplets.append((h_path, e_path, m_path))

            if (idx + 1) % 50 == 0 or idx == len(wav_files) - 1:
                elapsed = time.time() - start_time
                print(f"  Processed {idx + 1}/{len(wav_files)} files ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"  Warning: failed to process {wav_path}: {exc}")

    print(f"Cached {len(triplets)} feature triplets.")
    return triplets


def train_decoder(
    triplets: list[tuple[str, str, str]],
    output_dir: str = "checkpoints/acoustic_decoder",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_split: float = 0.1,
) -> str:
    """Train the Acoustic Decoder module."""
    os.makedirs(output_dir, exist_ok=True)

    # Train / Val split
    np.random.seed(42)
    indices = np.random.permutation(len(triplets))
    val_size = max(1, int(len(triplets) * val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_triplets = [triplets[i] for i in train_indices]
    val_triplets = [triplets[i] for i in val_indices]

    train_loader = DataLoader(
        SpeechMelDataset(train_triplets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_speech_mel,
        pin_memory=(device.startswith("cuda")),
    )
    val_loader = DataLoader(
        SpeechMelDataset(val_triplets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_speech_mel,
    )

    print(f"\nModel & Training Setup:")
    print(f"  Train set: {len(train_triplets)} samples")
    print(f"  Val set:   {len(val_triplets)} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs:    {epochs}")
    print(f"  Device:    {device}")

    model = AcousticDecoder(content_dim=768, speaker_dim=192, mel_dim=80, hidden_dim=512, n_conv_blocks=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss(reduction="none")

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best.pt")
    latest_checkpoint_path = os.path.join(output_dir, "latest.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_frames = 0

        for batch in train_loader:
            hubert = batch["hubert"].to(device)
            ecapa = batch["ecapa"].to(device)
            mel_target = batch["mel"].to(device)
            mel_lens = batch["mel_lens"].to(device)

            max_target_len = mel_target.shape[2]
            # Predict mel
            mel_pred = model(hubert, ecapa, target_len=max_target_len)

            # Create mask for variable length sequences: (B, 1, T_mel)
            B, _, T_mel = mel_pred.shape
            mask = torch.arange(T_mel, device=device).unsqueeze(0).expand(B, -1) < mel_lens.unsqueeze(1)
            mask = mask.unsqueeze(1).float()  # (B, 1, T_mel)

            # Masked L1 loss
            loss_matrix = criterion(mel_pred, mel_target) * mask
            total_elements = mask.sum() * 80
            loss = loss_matrix.sum() / max(1.0, total_elements.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * B
            train_frames += B

        scheduler.step()
        epoch_train_loss = train_loss / max(1, train_frames)

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                hubert = batch["hubert"].to(device)
                ecapa = batch["ecapa"].to(device)
                mel_target = batch["mel"].to(device)
                mel_lens = batch["mel_lens"].to(device)

                max_target_len = mel_target.shape[2]
                mel_pred = model(hubert, ecapa, target_len=max_target_len)

                B, _, T_mel = mel_pred.shape
                mask = torch.arange(T_mel, device=device).unsqueeze(0).expand(B, -1) < mel_lens.unsqueeze(1)
                mask = mask.unsqueeze(1).float()

                loss_matrix = criterion(mel_pred, mel_target) * mask
                loss = loss_matrix.sum() / max(1.0, (mask.sum() * 80).item())
                val_loss += loss.item() * B
                val_count += B

        epoch_val_loss = val_loss / max(1, val_count)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train L1: {epoch_train_loss:.4f} | "
                f"Val L1: {epoch_val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Save checkpoint
        checkpoint_dict = {
            "epoch": epoch,
            "acoustic_decoder": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": epoch_val_loss,
            "train_loss": epoch_train_loss,
        }
        torch.save(checkpoint_dict, latest_checkpoint_path)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(checkpoint_dict, best_checkpoint_path)

    print(f"\nTraining Complete! Best Val L1 Loss: {best_val_loss:.4f}")
    print(f"Saved best checkpoint to: {best_checkpoint_path}")
    return best_checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Neural Acoustic Decoder.")
    parser.add_argument("--data-dir", required=True, help="Directory containing .wav speech files")
    parser.add_argument("--cache-dir", default="data/features", help="Directory to cache extracted features")
    parser.add_argument("--output-dir", default="checkpoints/acoustic_decoder", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    wav_files = glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"ERROR: No .wav files found in {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(wav_files)} .wav files in {args.data_dir}")
    triplets = extract_features(wav_files, args.cache_dir, device=args.device)
    if not triplets:
        print("ERROR: No valid audio files processed.", file=sys.stderr)
        return 1

    train_decoder(
        triplets=triplets,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
