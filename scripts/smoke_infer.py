"""Self-contained inference smoke test (no external data needed).

Builds a tiny 4-bar melody, loads a checkpoint, generates accompaniment, and
renders a WAV. Verifies the full inference stack (CPU torch + model + fluidsynth)
works — used to validate the Docker image / a fresh install.

    python scripts/smoke_infer.py --checkpoint checkpoints/best-epoch=007-val_loss=0.8431.ckpt
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Work whether the package is installed (Docker) or run from source.
try:
    from jam_transformer.config import load_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from jam_transformer.config import load_config

import numpy as np
import torch
from scipy.io import wavfile

from dataclasses import replace

from jam_transformer.tokenizer import build_tokenizer, NoteEvent
from jam_transformer.pipeline import load_checkpoint, generate_accompaniment
from jam_transformer.utils.midi_io import events_to_midi, midi_to_events
from jam_transformer.utils.audio import render_midi_to_wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best-epoch=007-val_loss=0.8431.ckpt")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="output/smoke")
    ap.add_argument("--melody", default="",
                    help="real melody MIDI to accompany (default: synthetic 4-bar motif)")
    ap.add_argument("--max_bars", type=int, default=0,
                    help="cap the input melody to its first N bars (0 = full)")
    args = ap.parse_args()

    cfg = load_config(args.config); cfg.model.compile = False
    tok = build_tokenizer(cfg.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] env: torch={torch.__version__}  device={device}")

    lit = load_checkpoint(args.checkpoint, cfg, tok.vocab_size); lit.eval(); lit.to(device)
    print(f"[2/5] checkpoint loaded: {Path(args.checkpoint).name}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mel_mid = Path(tempfile.mktemp(suffix=".mid"))
    if args.melody:
        events, mtempo = midi_to_events(Path(args.melody), cfg.tokenizer)
        mel = [e for e in events if e.track == "melody"]
        if not mel:                                   # no named melody track → use all
            mel = [replace(e, track="melody") for e in events]
        if args.max_bars > 0:
            mel = [e for e in mel if e.bar < args.max_bars]
        events_to_midi(mel, cfg.tokenizer, tempo_bpm=mtempo).dump(str(mel_mid))
        print(f"[3/5] loaded melody from {Path(args.melody).name}: {len(mel)} notes")
    else:
        motif = [60, 62, 64, 67]   # C D E G
        mel = [NoteEvent(track="melody", bar=b, position=i * 4, pitch=p, duration=4, velocity=80)
               for b in range(4) for i, p in enumerate(motif)]
        events_to_midi(mel, cfg.tokenizer, tempo_bpm=100).dump(str(mel_mid))
        print(f"[3/5] built {len(mel)}-note synthetic test melody (4 bars)")

    midi, tempo = generate_accompaniment(mel_mid, cfg, lit, tok, ["melody"])
    acc = sum(len(i.notes) for i in midi.instruments if (i.name or "").upper() != "MELODY")
    out_mid = out / "smoke.mid"; midi.dump(str(out_mid))
    print(f"[4/5] generated accompaniment notes: {acc}  (tempo={tempo:.0f})")

    out_wav = out / "smoke.wav"
    render_midi_to_wav(out_mid, out_wav, cfg.inference.soundfont, cfg.inference.sample_rate)
    peak = 0
    if out_wav.exists():
        _, x = wavfile.read(str(out_wav)); peak = int(np.max(np.abs(x.astype(np.int32))))
    print(f"[5/5] WAV render: {'peak=' + str(peak) if out_wav.exists() else 'skipped (no soundfont)'}")

    ok = acc > 0
    print("SMOKE PASS" if ok else "SMOKE FAIL: no accompaniment generated")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
