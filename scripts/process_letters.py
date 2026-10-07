"""Build glyph templates for the selected terminal cell size."""

import argparse

from console_art import (
    DEFAULT_CELL_HEIGHT,
    DEFAULT_CELL_WIDTH,
    LETTERS_DIR,
    ConsoleArt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH)
    parser.add_argument("--cell-height", type=int, default=DEFAULT_CELL_HEIGHT)
    parser.add_argument("--letters-dir", default=str(LETTERS_DIR))
    args = parser.parse_args()

    renderer = ConsoleArt(
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )
    lookup_path = renderer.process_letters(args.letters_dir)
    print(
        f"Built {args.cell_width}x{args.cell_height} glyph templates "
        f"at {lookup_path}"
    )


if __name__ == "__main__":
    main()
