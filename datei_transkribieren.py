"""
Audio-Datei offline transkribieren (mp3/wav/m4a/...)
Benutzung:
  python datei_transkribieren.py mein_audio.mp3
  python datei_transkribieren.py mein_audio.mp3 --sprache de --modell small
"""

import argparse
import sys
from pathlib import Path
from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(description="Audio-Datei -> Text (offline)")
    parser.add_argument("datei", help="Pfad zur Audiodatei")
    parser.add_argument("--sprache", default="auto", help="de, en, ... oder auto")
    parser.add_argument("--modell", default="base", help="tiny, base, small, medium")
    args = parser.parse_args()

    pfad = Path(args.datei)
    if not pfad.exists():
        print(f"Datei nicht gefunden: {pfad}")
        sys.exit(1)

    print(f"Lade Modell '{args.modell}'...")
    model = WhisperModel(args.modell, device="cpu", compute_type="int8")

    lang = None if args.sprache == "auto" else args.sprache
    segments, info = model.transcribe(str(pfad), language=lang, beam_size=5, vad_filter=True)

    print(f"Erkannte Sprache: {info.language} ({info.language_probability:.2f})")
    texte = []
    for seg in segments:
        print(f"[{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text}")
        texte.append(seg.text.strip())

    volltext = " ".join(texte)
    out = pfad.with_suffix(".txt")
    out.write_text(volltext, encoding="utf-8")
    print(f"\nFertig! Gespeichert in: {out.resolve()}")


if __name__ == "__main__":
    main()
