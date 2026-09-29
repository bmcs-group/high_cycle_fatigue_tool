"""
Render the HCFT icon from its SVG source into the raster formats needed
for the Windows build.

    hcft/resources/hcft_icon.svg  (source, edit this one)
        -> hcft/resources/hcft_icon.ico   multi-size icon for the exe and installer
        -> hcft/resources/hcft_icon.png   256 px window icon used at runtime
        -> build/windows/wizard_small_*.bmp   Inno Setup wizard header images

Every size is rendered directly from the SVG (no downscaling of one big
bitmap), so small sizes stay sharp. Uses only PyQt5 (QtSvg) and Pillow,
which are already dependencies of the tool.

Usage:  uv run python packaging/windows/make_icon.py
"""
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import QBuffer, QIODevice, Qt
from PyQt5.QtGui import QGuiApplication, QImage, QPainter
from PyQt5.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / 'hcft' / 'resources'
SVG_FILE = RESOURCES / 'hcft_icon.svg'
BUILD_DIR = ROOT / 'build' / 'windows'

ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
# Inno Setup picks the best match for the current DPI (100 % ... 250 %)
WIZARD_SMALL_SIZES = [55, 69, 83, 110, 138]


def render(renderer, size):
    """Render the SVG into a size x size RGBA Pillow image."""
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()

    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, 'PNG')
    return Image.open(BytesIO(bytes(buffer.data()))).convert('RGBA')


def main():
    app = QGuiApplication(sys.argv)  # noqa: F841 (needed by QPainter)
    renderer = QSvgRenderer(str(SVG_FILE))
    if not renderer.isValid():
        sys.exit(f'Could not read {SVG_FILE}')

    images = [render(renderer, size) for size in ICO_SIZES]
    largest = images[-1]
    largest.save(RESOURCES / 'hcft_icon.ico', format='ICO',
                 sizes=[(s, s) for s in ICO_SIZES], append_images=images[:-1])
    largest.save(RESOURCES / 'hcft_icon.png')

    # Wizard images: icon on white with a small margin (BMP has no alpha)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for size in WIZARD_SMALL_SIZES:
        icon = render(renderer, round(size * 0.9))
        canvas = Image.new('RGB', (size, size), 'white')
        offset = (size - icon.width) // 2
        canvas.paste(icon, (offset, offset), icon)
        canvas.save(BUILD_DIR / f'wizard_small_{size}.bmp')

    print(f'Icons written to {RESOURCES} and {BUILD_DIR}')


if __name__ == '__main__':
    main()
