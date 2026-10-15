# Create and activate venv
python -m venv .\venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas Pillow tqdm

# (Optional) Freeze versions
pip freeze > requirements.txt

Write-Host "Environment setup complete. Activate with: .\venv\Scripts\Activate.ps1"
