"""Quick script to download a small CMU ARCTIC subset for Acoustic Decoder training."""
import urllib.request
import os
import sys

BASE_URL = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_rms_arctic/wav/"
OUTPUT_DIR = "data/raw/cmu_arctic/rms"
# Download first 50 utterances (about 3 minutes of clean US-English speech)
N = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)
downloaded = 0

for i in range(1, N + 1):
    utt_id = f"arctic_a{i:04d}"
    out_path = os.path.join(OUTPUT_DIR, f"{utt_id}.wav")
    if os.path.isfile(out_path):
        downloaded += 1
        continue
    url = f"{BASE_URL}{utt_id}.wav"
    try:
        urllib.request.urlretrieve(url, out_path)
        downloaded += 1
        if i % 10 == 0 or i == N:
            print(f"  [{i}/{N}] Downloaded {utt_id}.wav")
    except Exception as e:
        print(f"  Warning: {utt_id} failed: {e}", file=sys.stderr)

print(f"\nDone. {downloaded}/{N} files in {OUTPUT_DIR}/")
