# console-display

## Demo

[![Watch the console-display video demo](https://img.youtube.com/vi/0Sk03KkoHuo/hqdefault.jpg)](https://youtu.be/0Sk03KkoHuo)

Click the thumbnail to watch the video.

## Class-based renderer

The newer entry points share the ConsoleArt class in scripts/console_art.py:

- scripts/process_letters.py builds glyph templates from the PNGs in letters/.
- scripts/image.py renders a still image as terminal text.
- scripts/video.py plays a video as terminal text.

The independent Display.py, render.py, pixel.py, fonts.py, and image_demo.py
scripts are in test_scripts/.

### Build templates and render

Build the default 3x6 glyph templates once:

    python scripts/process_letters.py --cell-width 3 --cell-height 6

Rebuild the lookup after changing template generation or polarity; the CSV
stores glyph strokes as high values (white ink on a black background).

Then render an image or play a video with the same cell dimensions:

    python scripts/image.py assets/capybara_contrast.png --cell-width 3 --cell-height 6
    python scripts/video.py assets/blinding_lights.mp4 --cell-width 3 --cell-height 6

Add `--color` to either renderer to color each glyph from its selected
template's weighted source pixels. The terminal background remains black.
For video audio, install FFmpeg with `ffplay` available on PATH and add
`--audio` to the video command.
During playback, press Space or Enter, or click in a mouse-reporting terminal,
to pause and resume. These controls are enabled by default; use `--no-controls`
to disable them. Pausing audio also requires the `psutil` dependency from
`requirements.txt`.

To use another size, rebuild the templates first and use that same size for
rendering. For example:

    python scripts/process_letters.py --cell-width 6 --cell-height 12
    python scripts/video.py assets/blinding_lights.mp4 --cell-width 6 --cell-height 12

Template samples and their size metadata are written as
`tables/lookup_<width>x<height>.csv` and the matching JSON file. For example,
3x6 templates use `tables/lookup_3x6.csv` and `tables/lookup_3x6.json`. Each
size gets its own lookup file, and image/video rendering selects the file for
its configured dimensions. The renderer checks the metadata and CSV columns
so a lookup generated for one size cannot silently be used with another.

Image and video rendering apply gamma correction by default (`--gamma 1.2`)
to darken highlights before glyph matching. Use `--gamma 1.0` to disable it,
or try nearby values to tune the character distribution.
Both renderers also use CLAHE on the grayscale frame before glyph matching
(`--clahe-clip-limit 2.0`). Use `--no-clahe` to turn it off. CLAHE affects
glyph selection; color output is still sampled from the original frame.

The legacy renderers use the fixed 3x6 file `tables/lookup_3x6.csv`. The old
`lookup.csv`, `lookup_out.csv`, and `lookup_out_sorted.csv` files are no longer
used. The legacy `render.py` saves its processed image to
`outputs/output_image.jpg`.
