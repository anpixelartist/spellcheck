import os
import requests
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_FILE = CORPUS_DIR / "training.txt"

CORPUS_URLS = [
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
]

def download_corpus(progress_callback=None) -> str:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_FILE.exists():
        if progress_callback:
            progress_callback("Corpus already cached. Delete corpus/training.txt to re-download.")
        return str(CORPUS_FILE)

    total = len(CORPUS_URLS)
    all_text = []

    for i, url in enumerate(CORPUS_URLS):
        if progress_callback:
            progress_callback(f"Downloading corpus {i+1}/{total}...")

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            text = resp.text

            start = text.find("*** START OF" if "*** START OF" in text else "***START")
            end = text.find("*** END OF" if "*** END OF" in text else "***END")

            if start != -1 and end != -1:
                text = text[start:end]
            elif start != -1:
                text = text[start:]

            all_text.append(text)

            if progress_callback:
                progress_callback(f"Downloaded {i+1}/{total} ({len(text):,} chars)")

        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Failed to download {url}: {e}")

    if not all_text:
        raise RuntimeError("Failed to download any corpus files. Check your internet connection.")

    combined = "\n\n".join(all_text)
    CORPUS_FILE.write_text(combined, encoding="utf-8")

    if progress_callback:
        progress_callback(f"Corpus saved: {len(combined):,} characters ({CORPUS_FILE})")

    return str(CORPUS_FILE)


def get_corpus_path() -> str | None:
    if CORPUS_FILE.exists():
        return str(CORPUS_FILE)
    return None


def get_corpus_stats() -> dict:
    if not CORPUS_FILE.exists():
        return {"exists": False, "size_chars": 0, "size_mb": 0}

    size = CORPUS_FILE.stat().st_size
    return {
        "exists": True,
        "size_chars": size,
        "size_mb": round(size / (1024 * 1024), 2),
        "path": str(CORPUS_FILE),
    }


def delete_corpus():
    if CORPUS_FILE.exists():
        CORPUS_FILE.unlink()
