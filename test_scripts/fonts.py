import fontforge
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
FONT_PATH = PROJECT_DIR / "assets" / "consola.ttf"
LETTERS_DIR = PROJECT_DIR / "letters"

F = fontforge.open(str(FONT_PATH))
for name in F:
    if F[name].isWorthOutputting() and F[name].unicode >= 0x20 and F[name].unicode <= 0x7a:
        filename = str(F[name].unicode) + ".png"
        # print name
        F[name].export(str(LETTERS_DIR / filename), 119)
        # F[name].export(filename, 600)     # set height to 600 pixels
