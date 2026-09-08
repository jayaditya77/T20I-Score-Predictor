import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def run(script):
    subprocess.run([sys.executable, str(BASE / script)], check=True, cwd=BASE)


if __name__ == "__main__":
    run("download_cricsheet.py")
    run("train_model.py")
    print("\nSetup complete. Start the app with:")
    print(f"{sys.executable} -m streamlit run app.py")
