@echo off
if not exist venv (python -m venv venv)
call venv\Scripts\activate.bat
pip install diffusers transformers translate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python main.py