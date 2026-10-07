"""Render a still image as terminal character art."""

import argparse
import os
import platform
import re
import sys

from console_art import ASSETS_DIR, DEFAULT_CELL_HEIGHT, DEFAULT_CELL_WIDTH, ConsoleArt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=str,
        default=str(ASSETS_DIR / "capybara_color.jpg"),
        help="image file to render",
    )
    parser.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH)
    parser.add_argument("--cell-height", type=int, default=DEFAULT_CELL_HEIGHT)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="disable adaptive histogram equalization",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.2,
        help="brightness curve before glyph matching (1.0 disables it)",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="color each glyph using its weighted source pixels",
    )
    parser.add_argument(
        "--color-depth",
        choices=("256", "24bit"),
        default="24bit",
        help="color palette used with --color (default: 24bit)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="leave the terminal at its current size",
    )
    args = parser.parse_args()

    renderer = ConsoleArt(
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        gamma=args.gamma,
    )
    text = renderer.render_image(
        args.image, color=args.color, color_depth=args.color_depth
    )
    rows = text.splitlines()

    if rows and not args.no_resize:
        visible_first_row = re.sub(r"\033\[[0-9;]*m", "", rows[0])
        columns = len(visible_first_row)
        lines = len(rows) + 1
        if platform.system() == "Windows":
            os.system(f"mode con: cols={columns} lines={lines}")
        else:
            sys.stdout.write(f"\033[8;{lines};{columns}t")
            sys.stdout.flush()

    print(text)


if __name__ == "__main__":
    main()
