@echo off
title Push-to-Talk Diktat (POS1 halten)
cd /d "%~dp0"
python push_to_talk.py --sprache auto --modell base
pause
