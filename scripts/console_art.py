"""Shared character-image and character-video rendering pipeline."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import cv2
import numpy as np

from terminal_controls import TerminalControls


PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_DIR / "assets"
LETTERS_DIR = PROJECT_DIR / "letters"
TABLES_DIR = PROJECT_DIR / "tables"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DEFAULT_CELL_WIDTH = 3
DEFAULT_CELL_HEIGHT = 6
DEFAULT_FIRST_CODE = 32
DEFAULT_LAST_CODE = 122
DEFAULT_LOOKUP = TABLES_DIR / f"lookup_{DEFAULT_CELL_WIDTH}x{DEFAULT_CELL_HEIGHT}.csv"

_ANSI16_RGB = np.array(
    [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
        (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ],
    dtype=np.int32,
)
_ANSI256_CUBE_LEVELS = np.array((0, 95, 135, 175, 215, 255), dtype=np.int32)


def _rgb_to_ansi256(rgb_colors: np.ndarray) -> np.ndarray:
    """Map RGB colors to the nearest standard ANSI 256-color palette entry."""
    rgb = np.asarray(rgb_colors, dtype=np.int32)

    errors_16 = np.sum((rgb[:, np.newaxis, :] - _ANSI16_RGB) ** 2, axis=2)
    best_indices = np.argmin(errors_16, axis=1).astype(np.int32)
    best_errors = errors_16[np.arange(len(rgb)), best_indices]

    cube_levels = np.digitize(rgb, (48, 115, 155, 195, 235))
    cube_rgb = _ANSI256_CUBE_LEVELS[cube_levels]
    cube_errors = np.sum((rgb - cube_rgb) ** 2, axis=1)
    cube_indices = (
        16 + 36 * cube_levels[:, 0] + 6 * cube_levels[:, 1] + cube_levels[:, 2]
    )
    use_cube = cube_errors < best_errors
    best_indices[use_cube] = cube_indices[use_cube]
    best_errors[use_cube] = cube_errors[use_cube]

    gray_steps = np.clip(np.rint((rgb.mean(axis=1) - 8) / 10), 0, 23).astype(np.int32)
    gray_values = 8 + 10 * gray_steps
    gray_errors = np.sum((rgb - gray_values[:, np.newaxis]) ** 2, axis=1)
    use_gray = gray_errors < best_errors
    best_indices[use_gray] = 232 + gray_steps[use_gray]
    return best_indices


_SCORE_GLYPHS = cp.RawKernel(
    r"""
    extern "C" __global__
    void score_glyphs(
        const int* image,
        const int* glyphs,
        int* scores,
        const int cell_width,
        const int cell_height,
        const int window_width,
        const int glyph_count,
        const int brightness_offset
    ) {
        const int cell_index = blockIdx.x;
        const int glyph_index = threadIdx.x;
        if (glyph_index >= glyph_count) return;

        const int cell_x = cell_index % window_width;
        const int cell_y = cell_index / window_width;
        const int image_width = window_width * cell_width;
        const int image_base =
            cell_y * cell_height * image_width + cell_x * cell_width;
        const int pixels_per_glyph = cell_width * cell_height;

        int score = 0;
        for (int y = 0; y < cell_height; ++y) {
            for (int x = 0; x < cell_width; ++x) {
                const int pixel = y * cell_width + x;
                const int image_value = image[image_base + y * image_width + x];
                const int glyph_value =
                    glyphs[glyph_index * pixels_per_glyph + pixel];
                const int difference =
                    image_value - glyph_value - brightness_offset;
                score += difference * difference;
            }
        }
        scores[cell_index * glyph_count + glyph_index] = score;
    }
    """,
    "score_glyphs",
)

_CHOOSE_GLYPHS = cp.RawKernel(
    r"""
    extern "C" __global__
    void choose_glyphs(
        const int* scores,
        const int* glyph_codes,
        unsigned char* output,
        int* selected_indices,
        const int output_stride,
        const int glyph_count
    ) {
        const int cell_index = blockIdx.x;
        if (threadIdx.x != 0) return;

        int best_index = 0;
        int best_score = scores[cell_index * glyph_count];
        for (int i = 1; i < glyph_count; ++i) {
            const int score = scores[cell_index * glyph_count + i];
            if (score < best_score) {
                best_score = score;
                best_index = i;
            }
        }
        output[cell_index * output_stride] =
            (unsigned char)glyph_codes[best_index];
        selected_indices[cell_index] = best_index;
    }
    """,
    "choose_glyphs",
)

_COLOR_GLYPHS = cp.RawKernel(
    r"""
    extern "C" __global__
    void color_glyphs(
        const unsigned char* frame,
        const int* glyphs,
        const int* selected_indices,
        unsigned char* output,
        const int cell_width,
        const int cell_height,
        const int window_width,
        const int image_width
    ) {
        const int cell_index = blockIdx.x;
        if (threadIdx.x != 0) return;

        const int cell_x = cell_index % window_width;
        const int cell_y = cell_index / window_width;
        const int glyph_index = selected_indices[cell_index];
        const int pixels_per_glyph = cell_width * cell_height;
        int weight_sum = 0;
        int red_sum = 0;
        int green_sum = 0;
        int blue_sum = 0;

        for (int y = 0; y < cell_height; ++y) {
            for (int x = 0; x < cell_width; ++x) {
                const int pixel = y * cell_width + x;
                const int weight = glyphs[glyph_index * pixels_per_glyph + pixel];
                const int image_offset =
                    ((cell_y * cell_height + y) * image_width
                    + cell_x * cell_width + x) * 3;
                weight_sum += weight;
                blue_sum += weight * frame[image_offset];
                green_sum += weight * frame[image_offset + 1];
                red_sum += weight * frame[image_offset + 2];
            }
        }

        if (weight_sum == 0) {
            output[cell_index * 4 + 1] = 0;
            output[cell_index * 4 + 2] = 0;
            output[cell_index * 4 + 3] = 0;
        } else {
            output[cell_index * 4 + 1] = (unsigned char)((red_sum + weight_sum / 2) / weight_sum);
            output[cell_index * 4 + 2] = (unsigned char)((green_sum + weight_sum / 2) / weight_sum);
            output[cell_index * 4 + 3] = (unsigned char)((blue_sum + weight_sum / 2) / weight_sum);
        }
    }
    """,
    "color_glyphs",
)


class ConsoleArt:
    """Build glyph templates and render still images or videos as terminal art.

    The cell size is the number of samples stored for each glyph. Whenever it
    changes, call process_letters() to rebuild the matching lookup file.
    """

    def __init__(
        self,
        cell_width: int = DEFAULT_CELL_WIDTH,
        cell_height: int = DEFAULT_CELL_HEIGHT,
        *,
        first_code: int = DEFAULT_FIRST_CODE,
        last_code: int = DEFAULT_LAST_CODE,
        brightness_offset: int = 0,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        gamma: float = 1.2,
        lookup_path: Path | None = None,
    ) -> None:
        if cell_width <= 0 or cell_height <= 0:
            raise ValueError("Cell width and height must both be positive.")
        if first_code > last_code:
            raise ValueError("first_code must be less than or equal to last_code.")

        self.cell_width = cell_width
        self.cell_height = cell_height
        self.first_code = first_code
        self.last_code = last_code
        self.brightness_offset = brightness_offset
        if clahe and clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be greater than zero.")
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid,
            )
            if clahe
            else None
        )
        if gamma <= 0:
            raise ValueError("gamma must be greater than zero.")
        self.gamma = gamma
        gamma_values = np.arange(256, dtype=np.float32) / 255.0
        self._gamma_table = np.clip(
            np.round((gamma_values ** gamma) * 255.0), 0, 255
        ).astype(np.uint8)
        self.lookup_path = (
            Path(lookup_path)
            if lookup_path is not None
            else TABLES_DIR / f"lookup_{cell_width}x{cell_height}.csv"
        )
        self.metadata_path = self.lookup_path.with_suffix(".json")
        self._glyphs = None
        self._glyph_codes = None

    @property
    def pixel_count(self) -> int:
        return self.cell_width * self.cell_height

    def export_font(
        self,
        font_path: Path,
        letters_dir: Path = LETTERS_DIR,
        *,
        pixel_size: int = 119,
    ) -> list[Path]:
        """Export supported glyphs from a FontForge font as numbered PNGs.

        FontForge is imported here so rendering does not require it unless this
        font-export operation is used.
        """
        if pixel_size <= 0:
            raise ValueError("pixel_size must be greater than zero.")

        try:
            import fontforge
        except ImportError as error:
            raise RuntimeError(
                "Font export requires FontForge's Python module. Run this "
                "method with the Python environment provided by FontForge."
            ) from error

        font_path = Path(font_path)
        letters_dir = Path(letters_dir)
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        letters_dir.mkdir(parents=True, exist_ok=True)

        font = fontforge.open(str(font_path.resolve()))
        exported_paths = []
        try:
            for glyph_name in font:
                glyph = font[glyph_name]
                code = glyph.unicode
                if (
                    glyph.isWorthOutputting()
                    and self.first_code <= code <= self.last_code
                ):
                    output_path = letters_dir / f"{code}.png"
                    glyph.export(str(output_path), pixel_size)
                    exported_paths.append(output_path)
        finally:
            font.close()

        if not exported_paths:
            raise ValueError(
                f"No exportable glyphs in {font_path} for codes "
                f"{self.first_code} through {self.last_code}."
            )
        return exported_paths

    def process_letters(self, letters_dir: Path = LETTERS_DIR) -> Path:
        """Resize exported glyph PNGs into templates for this cell size."""
        letters_dir = Path(letters_dir)
        rows = []
        for image_path in sorted(
            letters_dir.glob("*.png"),
            key=lambda path: (
                (0, int(path.stem))
                if path.stem.isdigit()
                else (1, path.stem)
            ),
        ):
            try:
                code = int(image_path.stem)
            except ValueError:
                continue
            if not self.first_code <= code <= self.last_code:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read glyph image: {image_path}")
            interpolation = (
                cv2.INTER_AREA
                if self.cell_width <= image.shape[1]
                and self.cell_height <= image.shape[0]
                else cv2.INTER_CUBIC
            )
            sample = cv2.resize(
                image,
                (self.cell_width, self.cell_height),
                interpolation=interpolation,
            )
            # Store glyph ink as high values (white) and its background as low
            # values (black), matching the convention used by the CUDA kernels.
            rows.append((code, (255 - sample).reshape(-1).tolist()))

        if not rows:
            raise ValueError(
                f"No numeric glyph PNGs in {letters_dir} for codes "
                f"{self.first_code} through {self.last_code}."
            )

        self.lookup_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["unicode"] + [str(i) for i in range(self.pixel_count)]
        with self.lookup_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for code, samples in rows:
                writer.writerow([code, *samples])

        metadata = {
            "cell_width": self.cell_width,
            "cell_height": self.cell_height,
            "first_code": self.first_code,
            "last_code": self.last_code,
            "glyph_codes": [code for code, _ in rows],
            "glyph_polarity": "white-ink-on-black-background",
            "sample_method": "OpenCV INTER_AREA downscale / INTER_CUBIC upscale",
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        self._glyphs = None
        self._glyph_codes = None
        return self.lookup_path

    def _load_glyphs(self) -> None:
        if self._glyphs is not None:
            return
        if not self.lookup_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Glyph templates for {self.cell_width}x{self.cell_height} "
                f"were not found. Run process_letters.py with the same "
                f"cell-size arguments first."
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        stored_size = (
            metadata.get("cell_width"),
            metadata.get("cell_height"),
        )
        requested_size = (self.cell_width, self.cell_height)
        if stored_size != requested_size:
            raise ValueError(
                f"Lookup templates use {stored_size[0]}x{stored_size[1]}, "
                f"but this renderer uses {self.cell_width}x{self.cell_height}. "
                "Regenerate them with process_letters.py."
            )
        if metadata.get("glyph_polarity") != "white-ink-on-black-background":
            raise ValueError(
                "The lookup uses the old glyph polarity. Regenerate it with "
                "process_letters.py before rendering."
            )

        with self.lookup_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            expected_columns = {
                "unicode",
                *(f"{i}" for i in range(self.pixel_count)),
            }
            if set(reader.fieldnames or ()) != expected_columns:
                raise ValueError(
                    "The glyph CSV columns do not match its cell-size metadata. "
                    "Regenerate the templates with process_letters.py."
                )

            codes = []
            samples = []
            for row in reader:
                codes.append(int(row["unicode"]))
                samples.append(
                    [int(row[f"{i}"]) for i in range(self.pixel_count)]
                )

        if not codes:
            raise ValueError(f"No glyph templates found in {self.lookup_path}.")
        if len(codes) > 1024:
            raise ValueError("This CUDA matcher supports at most 1024 glyphs.")

        self._glyph_codes = cp.asarray(codes, dtype=cp.int32)
        self._glyphs = cp.asarray(samples, dtype=cp.int32)

    def frame_to_text(
        self,
        frame: np.ndarray,
        *,
        color: bool = False,
        color_depth: str = "24bit",
    ) -> str:
        """Convert one BGR or grayscale frame into newline-separated text."""
        if color and color_depth not in {"256", "24bit"}:
            raise ValueError("color_depth must be '256' or '24bit'.")
        self._load_glyphs()
        color_frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if self._clahe is not None:
            gray = self._clahe.apply(gray)

        # OPTIONAL GAMMA CORRECTION: gamma > 1 darkens bright and midtone
        # values before matching, which can reduce overuse of the brightest
        # glyphs. To remove this feature, delete this block and the gamma
        # setting/table setup in __init__ (plus --gamma in image.py/video.py).
        if self.gamma != 1.0:
            gray = cv2.LUT(gray, self._gamma_table)

        height = (gray.shape[0] // self.cell_height) * self.cell_height
        width = (gray.shape[1] // self.cell_width) * self.cell_width
        if width == 0 or height == 0:
            raise ValueError("The image is smaller than one configured cell.")

        window_width = width // self.cell_width
        window_height = height // self.cell_height
        cell_count = window_width * window_height
        image_gpu = cp.asarray(
            np.ascontiguousarray(gray[:height, :width]),
            dtype=cp.int32,
        )
        scores = cp.empty((cell_count, len(self._glyph_codes)), dtype=cp.int32)
        output_stride = 4 if color else 1
        output_shape = (cell_count, output_stride) if color else (cell_count,)
        output = cp.empty(output_shape, dtype=cp.uint8)
        selected_indices = cp.empty((cell_count,), dtype=cp.int32)

        _SCORE_GLYPHS(
            (cell_count,),
            (len(self._glyph_codes),),
            (
                image_gpu,
                self._glyphs,
                scores,
                self.cell_width,
                self.cell_height,
                window_width,
                len(self._glyph_codes),
                self.brightness_offset,
            ),
        )
        _CHOOSE_GLYPHS(
            (cell_count,),
            (1,),
            (
                scores,
                self._glyph_codes,
                output,
                selected_indices,
                output_stride,
                len(self._glyph_codes),
            ),
        )

        if color:
            cropped_color = np.ascontiguousarray(color_frame[:height, :width])
            frame_gpu = cp.asarray(cropped_color, dtype=cp.uint8)
            _COLOR_GLYPHS(
                (cell_count,),
                (1,),
                (
                    frame_gpu,
                    self._glyphs,
                    selected_indices,
                    output,
                    self.cell_width,
                    self.cell_height,
                    window_width,
                    width,
                ),
            )
            host_output = cp.asnumpy(output)
            flat_text = host_output[:, 0].tobytes().decode("ascii")
            rgb_colors = host_output[:, 1:4]
            palette_colors = (
                _rgb_to_ansi256(rgb_colors) if color_depth == "256" else None
            )
            colored_rows = []
            previous_color = None
            for row_index in range(window_height):
                row_start = row_index * window_width
                row_chars = flat_text[row_start:row_start + window_width]
                row_colors = rgb_colors[row_start:row_start + window_width]
                colored_row = [
                    "\033[48;5;0m" if color_depth == "256"
                    else "\033[48;2;0;0;0m"
                ]
                if color_depth == "256":
                    row_palette = palette_colors[
                        row_start:row_start + window_width
                    ]
                    for char, palette_index in zip(row_chars, row_palette):
                        current_color = int(palette_index)
                        if current_color != previous_color:
                            colored_row.append(
                                f"\033[38;5;{current_color}m"
                            )
                            previous_color = current_color
                        colored_row.append(char)
                else:
                    for char, color_values in zip(row_chars, row_colors):
                        current_color = tuple(int(value) for value in color_values)
                        if current_color != previous_color:
                            red, green, blue = current_color
                            colored_row.append(
                                f"\033[38;2;{red};{green};{blue}m"
                            )
                            previous_color = current_color
                        colored_row.append(char)
                colored_rows.append("".join(colored_row))
            return "\n".join(colored_rows) + "\033[0m"

        flat_text = cp.asnumpy(output).tobytes().decode("ascii")
        return "\n".join(
            flat_text[row * window_width:(row + 1) * window_width]
            for row in range(window_height)
        )

    def render_image(
        self,
        image_path: Path | str,
        *,
        color: bool = False,
        color_depth: str = "24bit",
    ) -> str:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.frame_to_text(
            image, color=color, color_depth=color_depth
        )

    def play_video(
        self,
        video_path: Path | str,
        *,
        resize_terminal: bool = True,
        color: bool = False,
        color_depth: str = "24bit",
        audio: bool = False,
        controls: bool = True,
    ) -> None:
        """Render a video frame by frame, synchronized to its reported FPS."""
        audio_process = None
        audio_controller = None
        if audio:
            import psutil

            ffplay_path = shutil.which("ffplay")
            if ffplay_path is None:
                raise FileNotFoundError(
                    "Audio playback requires ffplay (from FFmpeg) on PATH."
                )

        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_rate = float(video.get(cv2.CAP_PROP_FPS))
        frame_time = 1.0 / frame_rate if frame_rate > 0 else 1.0 / 30.0
        source_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        window_width = source_width // self.cell_width
        window_height = source_height // self.cell_height
        input_controls = TerminalControls() if controls else None
        terminal_started = False
        terminal_write_timings = []

        try:
            os.system("")
            if resize_terminal and window_width and window_height:
                sys.stdout.write(
                    f"\033[8;{window_height + 1};{window_width}t"
                )
            # Hide the cursor while ANSI positioning redraws each video row.
            sys.stdout.write("\033[?25l\033[2J")
            sys.stdout.flush()
            terminal_started = True

            if input_controls is not None:
                input_controls.__enter__()
            if audio:
                audio_process = subprocess.Popen(
                    [
                        ffplay_path,
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "error",
                        "-vn",
                        str(video_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                audio_controller = psutil.Process(audio_process.pid)
            playback_start = time.perf_counter()
            frame_index = 0
            paused = False
            pause_started = None

            def handle_pause_inputs() -> None:
                nonlocal paused, pause_started, playback_start
                if input_controls is None:
                    return
                for _ in range(input_controls.poll_toggles()):
                    if paused:
                        paused = False
                        playback_start += time.perf_counter() - pause_started
                        pause_started = None
                        if (
                            audio_controller is not None
                            and audio_process.poll() is None
                        ):
                            audio_controller.resume()
                    else:
                        paused = True
                        pause_started = time.perf_counter()
                        if (
                            audio_controller is not None
                            and audio_process.poll() is None
                        ):
                            audio_controller.suspend()

            while True:
                handle_pause_inputs()
                while paused:
                    time.sleep(0.01)
                    handle_pause_inputs()

                ok, frame = video.read()
                if not ok:
                    break
                text = self.frame_to_text(
                    frame, color=color, color_depth=color_depth
                )
                rows = text.splitlines()
                terminal_output = "".join(
                    f"\033[{row_index + 1};1H{row_text}"
                    for row_index, row_text in enumerate(rows)
                )
                write_started = time.perf_counter()
                sys.stdout.write(terminal_output)
                sys.stdout.flush()
                write_flush_ms = (time.perf_counter() - write_started) * 1000
                terminal_write_timings.append((frame_index + 1, write_flush_ms))
                frame_index += 1
                # Pace against the original playback timeline. Sleeping for
                # "one frame minus this iteration's work" can accumulate
                # sleep overshoot into noticeable drift over a long video.
                while True:
                    handle_pause_inputs()
                    if paused:
                        time.sleep(0.01)
                        continue
                    deadline = playback_start + frame_index * frame_time
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.01))
        finally:
            video.release()
            if audio_process is not None and audio_process.poll() is None:
                audio_process.terminate()
                try:
                    audio_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    audio_process.kill()
                    audio_process.wait()
            if terminal_started:
                # Clear the rendered frame, reset terminal colors, and restore
                # the cursor even if playback ends or is interrupted.
                sys.stdout.write("\033[0m\033[2J\033[H\033[?25h")
                sys.stdout.flush()
            if input_controls is not None:
                input_controls.__exit__(None, None, None)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            timing_path = OUTPUTS_DIR / "terminal_write_timings.csv"
            with timing_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(("frame", "write_flush_ms"))
                writer.writerows(terminal_write_timings)


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_CELL_HEIGHT",
    "DEFAULT_CELL_WIDTH",
    "DEFAULT_LOOKUP",
    "LETTERS_DIR",
    "OUTPUTS_DIR",
    "PROJECT_DIR",
    "TABLES_DIR",
    "ConsoleArt",
]
