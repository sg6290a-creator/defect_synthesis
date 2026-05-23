"""Gradio UI for Adversarial Defect Synthesis.

Run from custom_defect/ as the working directory:
    python -m ui.app
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

# Ensure imports work when launched as `python -m ui.app` from custom_defect/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.inference import synthesize_defects  # noqa: E402

RUNS_DIR = REPO_ROOT / "ui" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_TRAIN_DIR = REPO_ROOT / "data" / "demo_dataset" / "train"
DEMO_INPUT_DIR = REPO_ROOT / "data" / "demo_input"
SAM3_OUT_DIR = REPO_ROOT / "data" / "sam3_preprocessed"
SAM3_DEFAULT_CKPT = REPO_ROOT / "sam3.pt"

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


# -----------------------------------------------------------------------------
# Folder / file pickers (server-side tkinter; works because the app is local)
# -----------------------------------------------------------------------------


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _pick_zenity(kind: str, initial: str) -> str | None:
    bin_ = _which("zenity")
    if not bin_:
        return None
    args = [bin_, "--file-selection", "--title",
            "Select folder" if kind == "dir" else "Select file"]
    if kind == "dir":
        args.append("--directory")
    # Ensure trailing slash so zenity treats it as the initial directory
    init = initial if initial.endswith("/") else initial + "/"
    args.extend(["--filename", init])
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=600,
                             env={**os.environ})
        if out.returncode == 0:
            return out.stdout.strip()
        return ""  # user cancelled
    except Exception:  # noqa: BLE001
        return None


def _pick_kdialog(kind: str, initial: str) -> str | None:
    bin_ = _which("kdialog")
    if not bin_:
        return None
    if kind == "dir":
        args = [bin_, "--getexistingdirectory", initial]
    else:
        args = [bin_, "--getopenfilename", initial]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if out.returncode == 0:
            return out.stdout.strip()
        return ""
    except Exception:  # noqa: BLE001
        return None


def _pick_tk(kind: str, initial: str) -> str:
    if kind == "dir":
        code = (
            "import tkinter as tk, sys\n"
            "from tkinter import filedialog\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            f"p = filedialog.askdirectory(initialdir={initial!r}, title='Select folder')\n"
            "sys.stdout.write(p or '')\n"
        )
    else:
        code = (
            "import tkinter as tk, sys\n"
            "from tkinter import filedialog\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            f"p = filedialog.askopenfilename(initialdir={initial!r}, title='Select file')\n"
            "sys.stdout.write(p or '')\n"
        )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=600, env={**os.environ},
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"[picker error] {e}"


def _pick(kind: str, initial: str | None = None) -> str:
    """Open the best-available native dialog.

    Priority: zenity (GNOME) → kdialog (KDE) → tkinter fallback.
    Returns the chosen path, '' on cancel, or '[picker error] ...' on failure.
    """
    init = initial or os.path.expanduser("~")
    for fn in (_pick_zenity, _pick_kdialog):
        result = fn(kind, init)
        if result is not None:
            return result
    return _pick_tk(kind, init)


def pick_dir(current: str | None = None):
    chosen = _pick("dir", initial=current)
    return chosen or (current or "")


def pick_path(current: str | None = None):
    """Folder OR file. We open a folder dialog first; if user cancels we
    fall back to a file dialog. (Most users want a folder for Generate.)
    """
    chosen = _pick("dir", initial=current)
    if chosen:
        return chosen
    chosen = _pick("file", initial=current)
    return chosen or (current or "")


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------


def list_models() -> list[dict]:
    out = []
    if not RUNS_DIR.exists():
        return out
    for run in sorted(RUNS_DIR.iterdir()):
        if not run.is_dir():
            continue
        pt = run / f"{run.name}.pt"
        if not pt.exists():
            continue
        meta_path = run / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        out.append({
            "name": run.name,
            "pt": str(pt),
            "image_size": meta.get("image_size"),
            "epochs": meta.get("epochs_trained"),
            "created": meta.get("created"),
            "size_mb": round(pt.stat().st_size / 1024 / 1024, 1),
        })
    return out


def model_choices() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for m in list_models():
        label = f"{m['name']}  ({m['image_size']}px · {m['epochs']}ep · {m['size_mb']}MB)"
        items.append((label, m["pt"]))
    return items


def relink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=True)


# -----------------------------------------------------------------------------
# Training (streamed)
# -----------------------------------------------------------------------------


def _validate_train_inputs(name: str, a: str, b: str, m: str) -> str | None:
    if not name or not NAME_RE.match(name):
        return "모델 이름은 영문/숫자/_-./ (1-64자) 만 허용됩니다."
    for label, p in [("A", a), ("B", b), ("mask", m)]:
        if not p or not Path(p).is_dir():
            return f"{label} 폴더가 존재하지 않습니다: {p}"
    return None


def start_training(
    model_name: str,
    a_dir: str,
    b_dir: str,
    mask_dir: str,
    use_cuda: bool,
    epochs: int,
    batch_size: int,
    image_size: int,
    decay_epochs: int,
    save_freq_images: int,
    lr: float,
    save_every: int = 10,
):
    err = _validate_train_inputs(model_name, a_dir, b_dir, mask_dir)
    if err:
        yield err
        return

    epochs = int(epochs)
    decay_epochs = min(int(decay_epochs), epochs)
    run_dir = RUNS_DIR / model_name
    if run_dir.exists():
        for sub in ["data", "weights", "outputs", "logs"]:
            shutil.rmtree(run_dir / sub, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataroot = run_dir / "data"
    weights = run_dir / "weights"
    outputs = run_dir / "outputs"
    logs = run_dir / "logs"
    for d in [dataroot, weights, outputs, logs]:
        d.mkdir(parents=True, exist_ok=True)

    train_root = dataroot / model_name / "train"
    relink(train_root / "A", Path(a_dir))
    relink(train_root / "B", Path(b_dir))
    relink(train_root / "mask", Path(mask_dir))

    save_every = max(1, min(int(save_every), epochs))
    cmd = [
        sys.executable, "-u", "train.py",
        "--dataroot", str(dataroot),
        "--dataset", model_name,
        "--weightsf", str(weights),
        "--outf", str(outputs),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--image_size", str(image_size),
        "--decay_epochs", str(decay_epochs),
        "--save_freq", str(save_every),
        "--save_freq_images", str(save_freq_images),
        "--lr", str(lr),
    ]
    if use_cuda:
        cmd.insert(3, "--cuda")

    log_path = logs / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    log_buf: list[str] = [
        f"$ cd {REPO_ROOT}",
        "$ " + " ".join(cmd),
        "",
    ]
    yield "\n".join(log_buf)

    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
        env={
            **os.environ,
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )

    last_flush = time.time()
    with log_path.open("w") as lf:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            lf.write(line + "\n")
            log_buf.append(line)
            if len(log_buf) > 400:
                log_buf = log_buf[-400:]
            now = time.time()
            if now - last_flush > 0.5:
                last_flush = now
                yield "\n".join(log_buf)
        proc.wait()

    # Pick up the latest available checkpoint regardless of whether training
    # completed cleanly. With periodic --save_freq this gives the user
    # something usable even after a CUDA crash.
    cands = sorted(
        (weights / model_name).glob("netG_A2B_epoch_*.pth"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)),
    )
    if not cands:
        msg = "\n[FAIL] no netG_A2B checkpoint produced"
        if proc.returncode != 0:
            msg = f"\n[FAIL] train.py exited with code {proc.returncode}" + msg
        log_buf.append(msg)
        yield "\n".join(log_buf)
        return

    pth_src = cands[-1]
    reached_epoch = int(re.search(r"epoch_(\d+)", pth_src.name).group(1))
    pt_dst = run_dir / f"{model_name}.pt"
    shutil.copyfile(pth_src, pt_dst)
    meta = {
        "name": model_name,
        "image_size": int(image_size),
        "epochs_trained": reached_epoch,
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "created": datetime.now().isoformat(timespec="seconds"),
        "ngf": 64, "input_nc": 3, "output_nc": 3,
        "source_A": str(Path(a_dir).resolve()),
        "source_B": str(Path(b_dir).resolve()),
        "source_mask": str(Path(mask_dir).resolve()),
        "training_status": "ok" if proc.returncode == 0 else f"crashed (code {proc.returncode})",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if proc.returncode != 0:
        log_buf.append(
            f"\n[CRASH] train.py exited with code {proc.returncode}. "
            f"Recovered checkpoint at epoch {reached_epoch}/{epochs} → {pt_dst}"
        )
    else:
        log_buf.append(
            f"\n[OK] saved {pt_dst} (epoch {reached_epoch}, "
            f"{pt_dst.stat().st_size/1024/1024:.1f} MB)"
        )
    log_buf.append("→ Generate 탭의 ‘↻ refresh models’ 를 눌러 모델 목록을 갱신하세요.")
    yield "\n".join(log_buf)


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


def do_generate(pt_path: str, input_path: str, n: int, save_mask: bool, seed):
    if not pt_path:
        return [], "모델을 먼저 선택하세요."
    if not input_path or not Path(input_path).exists():
        return [], f"입력 경로가 없습니다: {input_path}"
    out_dir = Path(pt_path).parent / f"generated_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        results = synthesize_defects(
            pt_path=pt_path,
            input_path=input_path,
            out_dir=out_dir,
            n=int(n),
            save_mask=bool(save_mask),
            seed=int(seed) if seed not in (None, "", -1) else None,
        )
    except Exception as e:  # noqa: BLE001
        return [], f"[ERROR] {e}"

    gallery = []
    for r in results:
        gallery.append((r["fake"], f"fake: {Path(r['src']).name}"))
        if "mask" in r:
            gallery.append((r["mask"], f"mask: {Path(r['src']).name}"))
    return gallery, f"{len(results)} images → {out_dir}"


# -----------------------------------------------------------------------------
# Models tab
# -----------------------------------------------------------------------------


def models_table():
    rows = []
    for m in list_models():
        rows.append([
            m["name"], m["image_size"], m["epochs"], m["created"],
            f"{m['size_mb']} MB", m["pt"],
        ])
    return rows


def delete_model(name: str) -> str:
    if not name or not NAME_RE.match(name):
        return "유효한 이름을 입력하세요."
    target = RUNS_DIR / name
    if not target.exists():
        return f"존재하지 않습니다: {target}"
    shutil.rmtree(target)
    return f"삭제됨: {target}"


# -----------------------------------------------------------------------------
# SAM 3 preprocessing
# -----------------------------------------------------------------------------


def run_sam3_preprocess(
    defect_dir: str,
    labels_dir: str,
    out_root: str,
    checkpoint: str,
    box_format: str,
    device: str,
    use_auto: bool,
    limit: int,
):
    if not defect_dir or not Path(defect_dir).is_dir():
        return "", "", f"결함 이미지 폴더가 없습니다: {defect_dir}", gr.update(), gr.update()
    if not checkpoint:
        return "", "", "SAM 3 체크포인트 경로가 필요합니다.", gr.update(), gr.update()
    if labels_dir and not Path(labels_dir).is_dir():
        return "", "", f"BB 라벨 폴더가 없습니다: {labels_dir}", gr.update(), gr.update()
    if not labels_dir and not use_auto:
        return "", "", "BB 라벨 폴더를 넣거나 Auto mask를 켜세요.", gr.update(), gr.update()

    out_root_path = Path(out_root or SAM3_OUT_DIR)
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "sam3_make_masks.py"),
        "--defect-dir",
        defect_dir,
        "--out-root",
        str(out_root_path),
        "--checkpoint",
        checkpoint,
        "--box-format",
        box_format,
        "--device",
        device,
    ]
    if labels_dir:
        cmd.extend(["--labels-dir", labels_dir])
    if use_auto:
        cmd.append("--auto")
    if int(limit) > 0:
        cmd.extend(["--limit", str(int(limit))])

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log = "$ " + " ".join(cmd) + "\n\n" + proc.stdout
    if proc.stderr:
        log += "\n[stderr]\n" + proc.stderr

    b_dir = out_root_path / "train" / "B"
    mask_dir = out_root_path / "train" / "mask"
    if proc.returncode != 0:
        return "", "", log, gr.update(), gr.update()
    return str(b_dir), str(mask_dir), log, str(b_dir), str(mask_dir)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------


def _muted_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Defect Synthesis", theme=_muted_theme()) as demo:
        gr.Markdown(
            "## Adversarial Defect Synthesis\n"
            "정상 이미지(A) ↔ 결함 이미지(B) + 마스크로 CycleGAN을 학습하고, 정상 이미지에서 결함 이미지를 생성합니다."
        )

        with gr.Tab("Preprocess"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        sam3_defect_dir = gr.Textbox(label="결함 이미지 폴더", scale=4)
                        sam3_defect_btn = gr.Button("📁", scale=1, min_width=40)
                    with gr.Row():
                        sam3_labels_dir = gr.Textbox(label="BB 라벨 폴더 (선택, YOLO txt 기본)", scale=4)
                        sam3_labels_btn = gr.Button("📁", scale=1, min_width=40)
                    with gr.Row():
                        sam3_out_root = gr.Textbox(
                            label="출력 dataset root",
                            value=str(SAM3_OUT_DIR),
                            scale=4,
                        )
                        sam3_out_btn = gr.Button("📁", scale=1, min_width=40)
                    with gr.Row():
                        sam3_checkpoint = gr.Textbox(
                            label="SAM 3 체크포인트",
                            value=str(SAM3_DEFAULT_CKPT) if SAM3_DEFAULT_CKPT.exists() else "sam3.pt",
                            placeholder="e.g. sam3.pt",
                            scale=4,
                        )
                        sam3_checkpoint_btn = gr.Button("📄", scale=1, min_width=40)
                    with gr.Row():
                        sam3_device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
                        sam3_box_format = gr.Dropdown(
                            ["yolo", "xyxy", "coco"],
                            value="yolo",
                            label="BB format",
                        )
                    with gr.Row():
                        sam3_auto = gr.Checkbox(value=False, label="BB 없으면 Auto mask 사용")
                        sam3_limit = gr.Number(value=0, precision=0, label="처리 개수 제한 (0=전체)")
                    sam3_run_btn = gr.Button("Create Masks (SAM 3)", variant="primary")
                with gr.Column(scale=1):
                    sam3_b_out = gr.Textbox(label="생성된 B 폴더", interactive=False)
                    sam3_mask_out = gr.Textbox(label="생성된 Mask 폴더", interactive=False)
                    sam3_log = gr.Textbox(
                        label="Preprocess log",
                        lines=22,
                        max_lines=22,
                        interactive=False,
                    )

            sam3_defect_btn.click(pick_dir, [sam3_defect_dir], [sam3_defect_dir])
            sam3_labels_btn.click(pick_dir, [sam3_labels_dir], [sam3_labels_dir])
            sam3_out_btn.click(pick_dir, [sam3_out_root], [sam3_out_root])
            sam3_checkpoint_btn.click(pick_path, [sam3_checkpoint], [sam3_checkpoint])

        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column(scale=1):
                    name = gr.Textbox(
                        label="모델 이름 (.pt 파일명)",
                        value="demo_toy",
                        placeholder="e.g. corrosion_v1",
                    )
                    with gr.Row():
                        a_in = gr.Textbox(
                            label="A 폴더 (정상 이미지)",
                            value=str(DEMO_TRAIN_DIR / "A"),
                            scale=4,
                        )
                        a_btn = gr.Button("📁", scale=1, min_width=40)
                    with gr.Row():
                        b_in = gr.Textbox(
                            label="B 폴더 (결함 이미지)",
                            value=str(DEMO_TRAIN_DIR / "B"),
                            scale=4,
                        )
                        b_btn = gr.Button("📁", scale=1, min_width=40)
                    with gr.Row():
                        m_in = gr.Textbox(
                            label="Mask 폴더 (결함 마스크)",
                            value=str(DEMO_TRAIN_DIR / "mask"),
                            scale=4,
                        )
                        m_btn = gr.Button("📁", scale=1, min_width=40)
                    use_cuda = gr.Checkbox(value=False, label="CUDA 사용")
                    with gr.Row():
                        epochs = gr.Slider(1, 400, value=2, step=1, label="Epochs")
                        decay = gr.Slider(0, 400, value=1, step=1, label="Decay start")
                    with gr.Row():
                        bs = gr.Slider(1, 8, value=1, step=1, label="Batch size")
                        size = gr.Dropdown([64, 96, 128, 192, 256, 320], value=128, label="Image size")
                    with gr.Row():
                        save_every = gr.Slider(1, 100, value=10, step=1, label="Checkpoint every N epochs")
                        save_freq_img = gr.Slider(10, 500, value=100, step=10, label="Sample save interval")
                    lr = gr.Number(value=0.0002, label="Learning rate", precision=6)
                    start_btn = gr.Button("Start Training", variant="primary")
                with gr.Column(scale=1):
                    log_box = gr.Textbox(
                        label="Training log",
                        lines=24, max_lines=24, autoscroll=True,
                        interactive=False,
                    )
                    gr.Markdown(
                        "> 학습은 길게 걸립니다. 종료 후 자동으로 `runs/{name}/{name}.pt` 가 만들어집니다."
                    )
            a_btn.click(pick_dir, [a_in], [a_in])
            b_btn.click(pick_dir, [b_in], [b_in])
            m_btn.click(pick_dir, [m_in], [m_in])
            start_btn.click(
                start_training,
                inputs=[name, a_in, b_in, m_in, use_cuda, epochs, bs, size, decay, save_freq_img, lr, save_every],
                outputs=[log_box],
            )
            sam3_run_btn.click(
                run_sam3_preprocess,
                inputs=[
                    sam3_defect_dir,
                    sam3_labels_dir,
                    sam3_out_root,
                    sam3_checkpoint,
                    sam3_box_format,
                    sam3_device,
                    sam3_auto,
                    sam3_limit,
                ],
                outputs=[sam3_b_out, sam3_mask_out, sam3_log, b_in, m_in],
            )

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(
                        choices=model_choices(),
                        label="모델 (.pt)",
                        interactive=True,
                    )
                    refresh = gr.Button("↻ refresh models", size="sm")
                    with gr.Row():
                        inp = gr.Textbox(
                            label="입력 (정상 이미지 폴더 또는 단일 파일)",
                            value=str(DEMO_INPUT_DIR),
                            scale=4,
                        )
                        inp_btn = gr.Button("📁", scale=1, min_width=40)
                    n_out = gr.Slider(1, 200, value=8, step=1, label="생성 개수")
                    save_mask = gr.Checkbox(value=True, label="결함 마스크도 저장")
                    seed = gr.Number(value=-1, label="seed (-1 = 랜덤)", precision=0)
                    gen_btn = gr.Button("Generate", variant="primary")
                    out_msg = gr.Textbox(label="status", interactive=False)
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="results",
                        columns=3, rows=2, height=480,
                        preview=False, object_fit="contain",
                        show_label=True, show_share_button=False,
                        show_download_button=True, visible=False,
                    )
                    gallery_placeholder = gr.Markdown(
                        "_Generate 버튼을 누르면 결과 이미지가 여기에 표시됩니다._"
                    )

            def _on_generate(pt, inp_path, n, mask_flag, seed_val):
                images, msg = do_generate(pt, inp_path, n, mask_flag, seed_val)
                # Show gallery, hide placeholder
                return (
                    gr.update(value=images, visible=True),
                    gr.update(visible=False),
                    msg,
                )

            refresh.click(lambda: gr.update(choices=model_choices()), None, model_dd)
            inp_btn.click(pick_path, [inp], [inp])
            gen_btn.click(
                _on_generate,
                [model_dd, inp, n_out, save_mask, seed],
                [gallery, gallery_placeholder, out_msg],
            )

        with gr.Tab("Models"):
            tbl = gr.Dataframe(
                headers=["name", "image_size", "epochs", "created", "size", "pt_path"],
                value=models_table,
                interactive=False, label="trained models", wrap=True,
            )
            refresh2 = gr.Button("↻ refresh")
            with gr.Row():
                del_name = gr.Textbox(label="삭제할 모델 이름")
                del_btn = gr.Button("Delete", variant="stop")
            del_status = gr.Textbox(label="status", interactive=False)
            refresh2.click(models_table, None, tbl)
            del_btn.click(delete_model, del_name, del_status).then(models_table, None, tbl)

    return demo


def _find_free_port(start: int = 7860, end: int = 7900) -> int:
    import socket
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}-{end}")


if __name__ == "__main__":
    demo = build_ui()
    port = int(os.environ.get("GRADIO_SERVER_PORT") or _find_free_port())
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
    )
