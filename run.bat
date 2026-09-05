@echo off
if not exist venv (python -m venv venv)
call venv\Scripts\activate.bat
pip install diffusers["torch"] torchvision transformers translate
python main.py