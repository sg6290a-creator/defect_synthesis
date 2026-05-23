"""Defect synthesis inference helper.

Loads a trained netG_A2B generator from a .pt checkpoint and produces
defective images (and predicted defect masks) from clean input images.
"""
from __future__ import annotations

import functools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Allow running from repo root by adding parent dir to sys.path before this import.
from src.models_mask import ResNetGenerator9


@dataclass
class ModelMeta:
    name: str
    image_size: int
    epochs_trained: int
    ngf: int = 64
    input_nc: int = 3
    output_nc: int = 3


def _norm_layer():
    return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)


def build_generator(meta: ModelMeta) -> ResNetGenerator9:
    return ResNetGenerator9(
        meta.input_nc,
        meta.output_nc,
        meta.ngf,
        norm_layer=_norm_layer(),
        use_dropout=False,
        n_blocks=9,
    )


def load_generator(pt_path: str | os.PathLike, device: str = "cuda") -> tuple[ResNetGenerator9, ModelMeta]:
    pt_path = Path(pt_path)
    meta_path = pt_path.with_suffix(".json")
    if not meta_path.exists():
        meta_path = pt_path.parent / "meta.json"
    if meta_path.exists():
        d = json.loads(meta_path.read_text())
        meta = ModelMeta(
            name=d.get("name", pt_path.stem),
            image_size=int(d.get("image_size", 256)),
            epochs_trained=int(d.get("epochs_trained", 0)),
            ngf=int(d.get("ngf", 64)),
            input_nc=int(d.get("input_nc", 3)),
            output_nc=int(d.get("output_nc", 3)),
        )
    else:
        meta = ModelMeta(name=pt_path.stem, image_size=256, epochs_trained=0)

    net = build_generator(meta)
    state = torch.load(str(pt_path), map_location=device)
    net.load_state_dict(state)
    net.eval().to(device)
    return net, meta


def _to_tensor(img: Image.Image, size: int, device: str) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # to [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def _from_tensor(t: torch.Tensor) -> Image.Image:
    t = t.detach().squeeze(0).cpu().clamp(-1, 1)
    arr = ((t * 0.5 + 0.5) * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def _mask_to_pil(m: torch.Tensor) -> Image.Image:
    m = m.detach().squeeze().cpu().clamp(0, 1)
    arr = (m.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def list_input_images(path: str | os.PathLike) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


@torch.no_grad()
def synthesize_defects(
    pt_path: str | os.PathLike,
    input_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    n: int = 8,
    save_mask: bool = True,
    seed: int | None = None,
    device: str | None = None,
) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_generator(pt_path, device=device)

    files = list_input_images(input_path)
    if not files:
        raise FileNotFoundError(f"No input images found at {input_path}")

    rng = random.Random(seed)
    if n <= 0:
        n = len(files)
    if n <= len(files):
        picks = rng.sample(files, n)
    else:
        picks = [rng.choice(files) for _ in range(n)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i, src in enumerate(picks):
        try:
            img = Image.open(src)
        except Exception as e:
            print(f"[skip] {src.name}: {e}")
            continue
        x = _to_tensor(img, meta.image_size, device)
        fake_img, fake_mask, _ = net(x)
        fake_pil = _from_tensor(fake_img)
        fake_name = f"{i:04d}_fake.png"
        fake_path = out_dir / fake_name
        fake_pil.save(fake_path)
        item = {"src": str(src), "fake": str(fake_path)}
        if save_mask:
            mask_pil = _mask_to_pil(fake_mask)
            mask_path = out_dir / f"{i:04d}_mask.png"
            mask_pil.save(mask_path)
            item["mask"] = str(mask_path)
        results.append(item)
    return results
