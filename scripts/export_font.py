"""Export glyph PNGs from any font supported by FontForge."""

import argparse

from console_art import (
    DEFAULT_FIRST_CODE,
    DEFAULT_LAST_CODE,
    LETTERS_DIR,
    ConsoleArt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "font", help="font file to export, such as assets/consolas.ttf"
    )
    parser.add_argument("--letters-dir", default=str(LETTERS_DIR))
    parser.add_argument("--first-code", type=int, default=DEFAULT_FIRST_CODE)
    parser.add_argument("--last-code", type=int, default=DEFAULT_LAST_CODE)
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=120,
        help="glyph export height in pixels (default: 120)",
    )
    args = parser.parse_args()

    renderer = ConsoleArt(first_code=args.first_code, last_code=args.last_code)
    exported = renderer.export_font(
        args.font,
        args.letters_dir,
        pixel_size=args.pixel_size,
    )
    print(f"Exported {len(exported)} glyphs to {args.letters_dir}")


if __name__ == "__main__":
    main()
