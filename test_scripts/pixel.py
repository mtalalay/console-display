import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from console_art import ConsoleArt


if __name__ == "__main__":
    lookup_path = ConsoleArt().process_letters()
    print(f"Built default 3x6 glyph templates at {lookup_path}")

