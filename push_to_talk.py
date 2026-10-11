"""
Push-to-Talk Diktat: POS1 / Home halten = aufnehmen, loslassen = direkt eintippen.
Offline (Whisper), mehrsprachig.

Benutzung:
  pip install -r requirements.txt
  python push_to_talk.py
  python push_to_talk.py --sprache de --modell base --enter

Ablauf:
  1. Skript starten (Modell wird einmal geladen, dauert 5-15 Sek.)
  2. In Discord / Word / Browser ins Textfeld klicken
  3. POS1 (Home) GEDRÜCKT HALTEN + sprechen
  4. LOSLASSEN -> Text wird transkribiert und per Strg+V direkt eingefügt
     Mit --enter wird danach automatisch Enter gedrückt (= wirklich "senden")

Tasten:
  Standard: home (POS1). Ändern mit --taste f8  (z.B. f8, f9, scroll lock)
  Beenden: ESC
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLERATE = 16000


def einfügen_als_tastatur(text, mit_enter=False):
    """Fügt Text per Zwischenablage + Strg+V in das aktive Fenster ein."""
    import pyperclip
    import keyboard as kb

    if not text.strip():
        print("(nichts erkannt, nichts eingefügt)")
        return

    pyperclip.copy(text)
    time.sleep(0.15)  # kurz warten, damit Loslassen der POS1-Taste registriert ist
    kb.press_and_release("ctrl+v")
    time.sleep(0.1)
    if mit_enter:
        kb.press_and_release("enter")
        print("Eingefügt + Enter gedrückt (= gesendet).")
    else:
        print("Eingefügt (ohne Enter).")


def main():
    parser = argparse.ArgumentParser(description="POS1 halten = sprechen, loslassen = senden")
    parser.add_argument("--sprache", default="auto", help="de, en, ... oder auto")
    parser.add_argument("--modell", default="base", help="tiny=am schnellsten, base=Standard, small=genauer")
    parser.add_argument("--taste", default="home", help="Taste zum Halten, Standard: home (=POS1). z.B. f8")
    parser.add_argument("--enter", action="store_true", help="Nach dem Einfügen automatisch Enter drücken")
    parser.add_argument("--min-sek", type=float, default=0.4, help="Aufnahmen kürzer als X Sek. ignorieren")
    args = parser.parse_args()

    try:
        import keyboard as kb
    except ImportError:
        print("Fehlt: pip install keyboard pyperclip")
        sys.exit(1)

    from faster_whisper import WhisperModel

    print(f"Lade Whisper-Modell '{args.modell}' (erster Start lädt es einmal, danach offline)...")
    model = WhisperModel(args.modell, device="cpu", compute_type="int8")
    lang = None if args.sprache == "auto" else args.sprache
    print("Modell bereit.")

    print(f"\n=== PUSH-TO-TALK BEREIT ===")
    print(f"Halte [{args.taste.upper()}] gedrückt zum Sprechen, loslassen zum Senden.")
    print(f"Enter-Modus: {'AN (wird direkt gesendet)' if args.enter else 'AUS (nur eintippen)'}")
    print("Beenden mit ESC.\n")

    aufnahme = []
    stream = None
    nimmt_auf = False
    start_zeit = 0
    lock = threading.Lock()

    def start_rec():
        nonlocal stream, aufnahme, nimmt_auf, start_zeit
        with lock:
            if nimmt_auf:
                return
            aufnahme = []
            start_zeit = time.time()

            def callback(indata, frames, t, status):
                if status:
                    print(status, file=sys.stderr)
                aufnahme.append(indata.copy())

            try:
                stream = sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype="float32", callback=callback)
                stream.start()
            except Exception as e:
                print(f"Mikro-Fehler: {e}")
                print("Tipp: Mikro in Windows-Einstellungen prüfen.")
                return
            nimmt_auf = True
            print("● Aufnahme... (sprechen, Taste halten)", flush=True)

    def stop_rec():
        nonlocal stream, nimmt_auf
        with lock:
            if not nimmt_auf:
                return
            dauert = time.time() - start_zeit
            nimmt_auf = False
            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            stream = None

        print(f"○ Stopp ({dauert:.1f}s) -> transkribiere...", flush=True)

        if dauert < args.min_sek or not aufnahme:
            print("(zu kurz, ignoriert)")
            return

        try:
            audio = np.concatenate(aufnahme, axis=0).flatten()
        except ValueError:
            print("(kein Audio)")
            return

        # zu leise? -> abbrechen
        if np.abs(audio).max() < 0.01:
            print("(zu leise / Stille, ignoriert)")
            return

        segments, info = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[{info.language}] {text}")
        if text:
            einfügen_als_tastatur(text, mit_enter=args.enter)
            # zusätzlich sichern
            Path("letzte_diktate.txt").open("a", encoding="utf-8").write(text + "\n")

    # POS1 als Push-to-Talk registrieren, Home-Tastendruck unterdrücken
    # damit der Cursor nicht springt, solange das Skript läuft.
    taste = args.taste.lower()
    kb.on_press_key(taste, lambda e: start_rec(), suppress=True)
    kb.on_release_key(taste, lambda e: stop_rec(), suppress=True)

    print("Warte auf Tastendruck...")
    kb.wait("esc")
    print("Beendet.")


if __name__ == "__main__":
    main()
