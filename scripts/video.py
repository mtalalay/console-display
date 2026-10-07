"""Play a video as terminal character art."""

import argparse

from console_art import ASSETS_DIR, DEFAULT_CELL_HEIGHT, DEFAULT_CELL_WIDTH, ConsoleArt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=str,
        default=str(ASSETS_DIR / "iron_man_trailer.mp4"),
        help="video file to play",
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
        default=1.8,
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
        "--audio",
        action="store_true",
        help="play the video's audio using ffplay (FFmpeg must be installed)",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="disable Space/Enter/click pause and resume controls",
    )
    parser.add_argument("--no-resize", action="store_true")
    args = parser.parse_args()

    renderer = ConsoleArt(
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        gamma=args.gamma,
    )
    renderer.play_video(
        args.video,
        resize_terminal=not args.no_resize,
        color=args.color,
        color_depth=args.color_depth,
        audio=args.audio,
        controls=not args.no_controls,
    )


if __name__ == "__main__":
    main()
