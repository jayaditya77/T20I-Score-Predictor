from pathlib import Path
import zipfile

import requests

URL = "https://cricsheet.org/downloads/t20s_json.zip"
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw"
ZIP_PATH = RAW_DIR / "t20s_json.zip"
EXTRACT_DIR = RAW_DIR / "t20s_json"


def safe_extract(zip_path: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            destination = (target_dir / member.filename).resolve()
            if target_root not in destination.parents and destination != target_root:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            zf.extract(member, target_dir)
            if member.filename.endswith(".json"):
                count += 1
    return count


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_json = list(EXTRACT_DIR.glob("*.json"))
    if existing_json:
        print(f"Cricsheet JSON already extracted: {len(existing_json):,} match files.")
        print("Skipping download. Delete data/raw/t20s_json if you want a fresh snapshot.")
        return

    print("Downloading official Cricsheet T20I JSON archive...")
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded / total:6.1%}", end="\r")

    print(f"\nSaved: {ZIP_PATH}")
    count = safe_extract(ZIP_PATH, EXTRACT_DIR)
    print(f"Extracted {count:,} JSON match files to {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
