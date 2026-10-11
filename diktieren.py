"""
Eigene Offline Speech-to-Text App zum Diktieren
Mehrsprachig (de/en/...) dank Whisper, automatische Spracherkennung.

Benutzung:
  pip install -r requirements.txt
  python diktieren.py
  python diktieren.py --sprache de   (erzwingt Deutsch)
  python diktieren.py --modell small  (genauer, aber langsamer)
"""

import argparse
import datetime
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

SAMPLERATE = 16000
CHANNELS = 1


def aufnehmen():
    """Nimmt vom Mikro auf bis Enter gedrückt wird. Gibt numpy-Audio zurück."""
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n=== Aufnahme läuft ===")
    print("Sprich jetzt. Drücke ENTER zum Stoppen.\n")

    aufnahme_chunks = []
    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype="float32", callback=callback):
        input()  # wartet auf Enter
        while not q.empty():
            aufnahme_chunks.append(q.get())

    # Rest aus Queue holen
    while not q.empty():
        aufnahme_chunks.append(q.get())

    if not aufnahme_chunks:
        print("Nichts aufgenommen (kein Mikro-Signal?).")
        sys.exit(1)

    audio = np.concatenate(aufnahme_chunks, axis=0).flatten()
    dauer = len(audio) / SAMPLERATE
    print(f"Aufgenommen: {dauer:.1f} Sekunden")
    return audio, dauer


def transkribieren(audio_path_or_array, modell_groesse="base", sprache=None):
    """Transkribiert per faster-whisper offline. sprache=None = automatisch erkennen."""
    print(f"Lade Modell '{modell_groesse}' (erster Start lädt ~150 MB, danach offline)...")
    # compute_type=int8 = schnell + wenig RAM, ideal für CPU
    model = WhisperModel(modell_groesse, device="cpu", compute_type="int8")

    # language=None -> automatische Erkennung (mehrsprachig)
    lang_param = None if (sprache in (None, "auto")) else sprache

    segments, info = model.transcribe(
        audio_path_or_array,
        language=lang_param,
        beam_size=5,
        vad_filter=True,  # filtert Stille automatisch
    )

    print(f"Erkannte Sprache: {info.language} (Sicherheit: {info.language_probability:.2f})")
    textteile = []
    for seg in segments:
        print(f"[{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text}")
        textteile.append(seg.text.strip())

    return " ".join(textteile), info.language


def main():
    parser = argparse.ArgumentParser(description="Offline Diktieren -> Text")
    parser.add_argument("--sprache", default="auto", help="de, en, fr, ... oder auto (Standard: auto)")
    parser.add_argument("--modell", default="base", help="tiny, base, small, medium (Standard: base)")
    parser.add_argument("--output", default=None, help="Zieldatei, z.B. notiz.txt")
    args = parser.parse_args()

    audio, _ = aufnehmen()

    # temporär als WAV speichern (faster-whisper mag Dateien oder numpy)
    tmp_wav = Path("aufnahme_tmp.wav")
    sf.write(str(tmp_wav), audio, SAMPLERATE)

    text, erkannte_sprache = transkribieren(str(tmp_wav), args.modell, args.sprache)

    print("\n=== ERGEBNIS ===")
    print(text if text else "(nichts erkannt)")
    print("==============\n")

    if args.output:
        ziel = Path(args.output)
    else:
        stempel = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ziel = Path(f"notiz_{stempel}.txt")

    ziel.write_text(text, encoding="utf-8")
    print(f"Gespeichert in: {ziel.resolve()}")

    # Aufräumen
    try:
        tmp_wav.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
