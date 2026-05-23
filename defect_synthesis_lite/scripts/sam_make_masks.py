"""Create defect masks from images with a SAM-family model.

The output layout matches the training code:

    out_root/train/B/<image>
    out_root/train/mask/<image-stem>.png

Use bbox labels when possible. Prompt-free automatic masks are supported, but
they are only a rough bootstrap because SAM does not know which segment is the
actual defect without a prompt.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def label_path_for(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def parse_boxes(label_path: Path, width: int, height: int, fmt: str) -> list[list[float]]:
    if not label_path.exists():
        return []

    boxes: list[list[float]] = []
    for raw in label_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        vals = [float(x) for x in raw.replace(",", " ").split()]

        if fmt == "yolo":
            if len(vals) < 5:
                raise ValueError(f"YOLO label needs class x y w h: {label_path}")
            _, cx, cy, bw, bh = vals[:5]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
        elif fmt == "xyxy":
            vals = vals[-4:] if len(vals) >= 5 else vals[:4]
            if len(vals) != 4:
                raise ValueError(f"xyxy label needs x1 y1 x2 y2: {label_path}")
            x1, y1, x2, y2 = vals
        elif fmt == "coco":
            vals = vals[-4:] if len(vals) >= 5 else vals[:4]
            if len(vals) != 4:
                raise ValueError(f"coco label needs x y w h: {label_path}")
            x1, y1, bw, bh = vals
            x2 = x1 + bw
            y2 = y1 + bh
        else:
            raise ValueError(f"Unsupported box format: {fmt}")

        boxes.append([
            max(0.0, min(float(width - 1), x1)),
            max(0.0, min(float(height - 1), y1)),
            max(0.0, min(float(width - 1), x2)),
            max(0.0, min(float(height - 1), y2)),
        ])
    return boxes


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def union_mask(masks: list[np.ndarray], width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        m = resize_mask(mask, width, height)
        out[m > 0] = 255
    return out


def load_ultralytics_model(model_path: str):
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise RuntimeError("Install ultralytics or use --backend meta.") from exc
    return SAM(model_path)


def ultralytics_mask(model, image_path: Path, boxes: list[list[float]], device: str) -> np.ndarray:
    img = read_image(image_path)
    height, width = img.shape[:2]
    kwargs = {"device": device, "verbose": False}
    if boxes:
        kwargs["bboxes"] = boxes
    results = model(str(image_path), **kwargs)
    if not results or results[0].masks is None:
        return np.zeros((height, width), dtype=np.uint8)
    masks = results[0].masks.data.detach().cpu().numpy()
    return union_mask([m > 0.5 for m in masks], width, height)


def load_meta_predictor(model_type: str, checkpoint: str, device: str):
    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError("Install segment-anything or use --backend ultralytics.") from exc

    if not checkpoint:
        raise RuntimeError("--checkpoint is required for --backend meta.")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam), SamAutomaticMaskGenerator(sam)


def meta_mask_from_boxes(predictor, image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    height, width = image.shape[:2]
    predictor.set_image(image)
    masks = []
    for box in boxes:
        pred_masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
        masks.append(pred_masks[int(np.argmax(scores))])
    return union_mask(masks, width, height)


def meta_mask_auto(generator, image: np.ndarray, min_area: int, max_area_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    max_area = int(width * height * max_area_ratio)
    masks = []
    for item in generator.generate(image):
        area = int(item.get("area", 0))
        if min_area <= area <= max_area:
            masks.append(item["segmentation"])
    return union_mask(masks, width, height)


def write_outputs(src: Path, mask: np.ndarray, b_dir: Path, mask_dir: Path) -> tuple[Path, Path]:
    b_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_out = b_dir / src.name
    mask_out = mask_dir / f"{src.stem}.png"
    shutil.copy2(src, image_out)
    Image.fromarray(mask.astype(np.uint8)).save(mask_out)
    return image_out, mask_out


def run(args: argparse.Namespace) -> int:
    defect_dir = args.defect_dir.resolve()
    out_root = args.out_root.resolve()
    b_dir = out_root / "train" / "B"
    mask_dir = out_root / "train" / "mask"
    labels_dir = args.labels_dir.resolve() if args.labels_dir else None

    images = list_images(defect_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {defect_dir}")

    if args.backend == "ultralytics":
        model = load_ultralytics_model(args.checkpoint)
        meta_predictor = None
        meta_auto = None
    else:
        model = None
        meta_predictor, meta_auto = load_meta_predictor(args.model_type, args.checkpoint, args.device)

    made = 0
    skipped = 0
    for image_path in images:
        image = read_image(image_path)
        height, width = image.shape[:2]
        boxes = []
        if labels_dir:
            boxes = parse_boxes(label_path_for(labels_dir, image_path), width, height, args.box_format)
            if not boxes and not args.auto:
                skipped += 1
                print(f"[skip] no bbox label: {image_path.name}")
                continue

        if args.backend == "ultralytics":
            mask = ultralytics_mask(model, image_path, boxes, args.device)
        elif boxes:
            mask = meta_mask_from_boxes(meta_predictor, image, boxes)
        else:
            if not args.auto:
                raise RuntimeError("Provide --labels-dir or enable --auto.")
            mask = meta_mask_auto(meta_auto, image, args.min_area, args.max_area_ratio)

        if mask.max() == 0:
            skipped += 1
            print(f"[skip] empty mask: {image_path.name}")
            continue

        image_out, mask_out = write_outputs(image_path, mask, b_dir, mask_dir)
        made += 1
        print(f"[ok] {image_path.name} -> {image_out.name}, {mask_out.name}")

    print(f"[done] masks={made}, skipped={skipped}")
    print(f"B_DIR={b_dir}")
    print(f"MASK_DIR={mask_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defect-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("data/sam_preprocessed"))
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--box-format", choices=["yolo", "xyxy", "coco"], default="yolo")
    parser.add_argument("--backend", choices=["ultralytics", "meta"], default="ultralytics")
    parser.add_argument("--checkpoint", required=True, help="SAM model path/name, e.g. sam_b.pt or sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b", help="Meta SAM model type: vit_b, vit_l, or vit_h")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--auto", action="store_true", help="Use prompt-free SAM masks when bbox labels are absent.")
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
