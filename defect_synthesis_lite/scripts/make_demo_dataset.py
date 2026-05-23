"""Create a tiny synthetic dataset for the local defect-synthesis demo."""
from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _metal_background(size: int, rng: random.Random, phase: float) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    base = 164 + 18 * np.sin((xx / size) * math.pi * 2.0 + phase)
    base += 10 * np.cos((yy / size) * math.pi * 3.0 + phase * 0.7)
    noise = np.array([[rng.gauss(0, 4) for _ in range(size)] for _ in range(size)])
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([arr + 8, arr + 4, arr], axis=-1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for x in range(size // 5, size, size // 4):
        draw.line((x, 0, x - size // 7, size), fill=(128, 132, 130), width=1)
    return img.filter(ImageFilter.GaussianBlur(radius=0.35))


def _add_defect(img: Image.Image, size: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    out = img.copy()
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(out)
    mask_draw = ImageDraw.Draw(mask)

    cx = rng.randint(size // 4, size * 3 // 4)
    cy = rng.randint(size // 4, size * 3 // 4)
    rx = rng.randint(size // 12, size // 5)
    ry = rng.randint(size // 18, size // 8)
    angle = rng.uniform(-0.8, 0.8)

    points = []
    for step in range(22):
        t = (step / 21) * math.pi * 2
        rj = 0.75 + rng.random() * 0.45
        x = math.cos(t) * rx * rj
        y = math.sin(t) * ry * rj
        xr = cx + math.cos(angle) * x - math.sin(angle) * y
        yr = cy + math.sin(angle) * x + math.cos(angle) * y
        points.append((xr, yr))

    defect_color = (
        rng.randint(62, 96),
        rng.randint(50, 76),
        rng.randint(42, 64),
    )
    draw.polygon(points, fill=defect_color)
    mask_draw.polygon(points, fill=255)

    for _ in range(3):
        x0 = cx + rng.randint(-rx, rx)
        y0 = cy + rng.randint(-ry, ry)
        x1 = x0 + rng.randint(-size // 5, size // 5)
        y1 = y0 + rng.randint(-size // 10, size // 10)
        draw.line((x0, y0, x1, y1), fill=(44, 40, 36), width=max(1, size // 64))
        mask_draw.line((x0, y0, x1, y1), fill=255, width=max(2, size // 48))

    return out.filter(ImageFilter.GaussianBlur(radius=0.2)), mask


def create_dataset(root: Path, count: int, size: int, seed: int) -> None:
    train_root = root / "data" / "demo_dataset" / "train"
    input_root = root / "data" / "demo_input"

    dirs = [
        train_root / "A",
        train_root / "B",
        train_root / "mask",
        input_root,
    ]
    for path in dirs:
        _reset_dir(path)

    rng = random.Random(seed)
    for idx in range(count):
        base = _metal_background(size, rng, phase=idx * 0.61)
        defect, mask = _add_defect(base, size, rng)

        name = f"demo_{idx:03d}.png"
        base.save(train_root / "A" / name)
        defect.save(train_root / "B" / name)
        mask.save(train_root / "mask" / name)

        if idx < max(3, count // 3):
            query = _metal_background(size, rng, phase=idx * 1.33 + 7)
            query.save(input_root / f"input_{idx:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    create_dataset(args.root.resolve(), args.count, args.size, args.seed)


if __name__ == "__main__":
    main()
