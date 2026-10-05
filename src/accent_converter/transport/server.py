"""
accent_converter.transport.server
=================================
Interactive Studio Web Server for live accent conversion testing.
Uses standard Python http.server (no extra dependencies required).
"""

from __future__ import annotations

import email
import io
import json
import os
import shutil
from email.message import Message
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
import soundfile as sf

from accent_converter.pipeline.offline import run_offline_pipeline

ROOT_DIR = Path(__file__).resolve().parents[3]
STATIC_INDEX = Path(__file__).resolve().parent / "index.html"
SCRATCH_DIR = ROOT_DIR / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SOURCE_WAV = ROOT_DIR / "datasets/l2arctic_release_v5.0/ASI/ASI/wav/arctic_a0001.wav"
DEFAULT_REF_RMS = ROOT_DIR / "datasets/extracted/cmu_arctic/cmu_us_rms_arctic/wav/arctic_a0001.wav"
DEFAULT_REF_SLT = ROOT_DIR / "datasets/extracted/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav"
CONFIG_PATH = ROOT_DIR / "configs/conformer_inference.yaml"


class StudioHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(STATIC_INDEX, "rb") as f:
                self.wfile.write(f.read())
            return

        if self.path.startswith("/api/sample/source"):
            if not DEFAULT_SOURCE_WAV.is_file():
                self.send_error(404, "Default source sample not found.")
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            with open(DEFAULT_SOURCE_WAV, "rb") as f:
                self.wfile.write(f.read())
            return

        if self.path.startswith("/api/sample/converted"):
            cached_out = SCRATCH_DIR / "test_converted.wav"
            if cached_out.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                with open(cached_out, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_error(404, "No converted sample ready yet.")
            return

        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/convert":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            speaker = "rms"
            target_accent = "us"
            audio_bytes = None

            # Parse multipart/form-data using standard email module
            if "multipart/form-data" in content_type:
                msg_bytes = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body_bytes
                msg = email.message_from_bytes(msg_bytes)
                if msg.is_multipart():
                    for part in msg.get_payload():
                        cd = part.get("Content-Disposition", "")
                        if 'name="speaker"' in cd:
                            speaker = part.get_payload(decode=True).decode("utf-8").strip()
                        elif 'name="target_accent"' in cd:
                            target_accent = part.get_payload(decode=True).decode("utf-8").strip()
                        elif 'name="audio"' in cd and part.get_payload(decode=True):
                            audio_bytes = part.get_payload(decode=True)

            input_wav_path = SCRATCH_DIR / "uploaded_input.wav"
            if audio_bytes:
                with open(input_wav_path, "wb") as f:
                    f.write(audio_bytes)
            else:
                input_wav_path = DEFAULT_SOURCE_WAV

            # 2. Speaker selection
            speaker = form.getvalue("speaker", "rms")
            if speaker == "slt" and DEFAULT_REF_SLT.is_file():
                ref_wav_path = DEFAULT_REF_SLT
            elif speaker == "source":
                ref_wav_path = input_wav_path
            else:
                ref_wav_path = DEFAULT_REF_RMS

            target_accent = form.getvalue("target_accent", "us")
            output_wav_path = SCRATCH_DIR / "studio_converted.wav"

            try:
                run_offline_pipeline(
                    audio_path=input_wav_path,
                    target_accent=target_accent,
                    enrollment_audio_path=ref_wav_path,
                    config_path=CONFIG_PATH,
                    output_path=output_wav_path,
                    device="cuda:0",
                )
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                with open(output_wav_path, "rb") as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Conversion error: {e}".encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def start_server(host: str = "127.0.0.1", port: int = 7860) -> None:
    server = HTTPServer((host, port), StudioHandler)
    print(f"\n=======================================================")
    print(f" Accent Converter Studio Running at: http://{host}:{port}")
    print(f"=======================================================\n")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
