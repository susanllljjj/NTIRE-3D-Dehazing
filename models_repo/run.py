import argparse
import os
import sys
import gc
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import load_image

# 让脚本在仓库根目录下可直接找到 src/pipeline_difix.py
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline_difix import DifixPipeline  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def get_resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


RESAMPLE = get_resample()


def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def build_ref_map(ref_dir: Path):
    ref_map = {}
    for p in list_images(ref_dir):
        ref_map[p.stem] = p
    return ref_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="本地模型目录，如 ./checkpoints/difix_hf")
    parser.add_argument("--input-dir", type=str, required=True, help="输入图片文件夹")
    parser.add_argument("--output-dir", type=str, required=True, help="输出图片文件夹")
    parser.add_argument("--ref-dir", type=str, default=None, help="参考图文件夹；文件名需与输入图同名")
    parser.add_argument("--prompt", type=str, default="remove degradation")
    parser.add_argument("--steps", type=int, default=1, help="num_inference_steps，官方示例默认 1")
    parser.add_argument("--timestep", type=int, default=199, help="官方示例默认 199")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="官方示例默认 0.0")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="送入模型前的最大边长；显存不够可改成 768 或 512；0 表示不缩放"
    )
    parser.add_argument(
        "--enable-xformers",
        action="store_true",
        help="如果环境已安装 xformers，则开启更省显存的 attention"
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="进一步省显存，但会更慢；启用后不要再手动 pipe.to(cuda)"
    )
    return parser.parse_args()


def resize_keep_ratio(image: Image.Image, max_side: int):
    if max_side is None or max_side <= 0:
        return image

    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), RESAMPLE)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main():
    args = parse_args()

    # 尽量减轻 CUDA 显存碎片问题
    #os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_dir = Path(args.model_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你指定了 --device cuda，但当前环境没有可用 CUDA")

    if ref_dir is not None and not ref_dir.exists():
        raise FileNotFoundError(f"参考图文件夹不存在: {ref_dir}")

    print(f"[INFO] loading model from: {model_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] max_side: {args.max_side}")
    print(f"[INFO] PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    if device == "cuda":
        # CUDA 推理优先用 fp16 降显存
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = DifixPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
    )

    # 常见省显存优化
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if args.enable_xformers and device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xformers memory efficient attention enabled")
        except Exception as e:
            print(f"[WARN] enable_xformers_memory_efficient_attention failed: {e}")

    if args.enable_cpu_offload and device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("[INFO] model cpu offload enabled")
        except Exception as e:
            print(f"[WARN] enable_model_cpu_offload failed, fallback to pipe.to(cuda): {e}")
            pipe.to(device)
    else:
        pipe.to(device)

    input_paths = list_images(input_dir)
    if not input_paths:
        raise FileNotFoundError(f"输入文件夹里没找到图片: {input_dir}")

    ref_map = build_ref_map(ref_dir) if ref_dir else None

    print(f"[INFO] found {len(input_paths)} input images")

    for idx, img_path in enumerate(input_paths, start=1):
        print(f"[{idx}/{len(input_paths)}] processing: {img_path.name}")

        input_image = None
        ref_image = None
        output_image = None

        try:
            input_image = load_image(str(img_path)).convert("RGB")
            orig_w, orig_h = input_image.size

            # 进模型前先缩小，减少显存
            proc_image = resize_keep_ratio(input_image, args.max_side)
            proc_w, proc_h = proc_image.size

            kwargs = dict(
                prompt=args.prompt,
                image=proc_image,
                num_inference_steps=args.steps,
                timesteps=[args.timestep],
                guidance_scale=args.guidance_scale,
            )

            if ref_map is not None:
                ref_path = ref_map.get(img_path.stem)
                if ref_path is None:
                    print(f"[WARN] 跳过 {img_path.name}，未找到同名参考图")
                    continue

                ref_image = load_image(str(ref_path)).convert("RGB")
                ref_image = ref_image.resize((proc_w, proc_h), RESAMPLE)
                kwargs["ref_image"] = ref_image

            with torch.inference_mode():
                output = pipe(**kwargs)
                output_image = output.images[0]

            # 强制恢复为输入图原始分辨率
            if output_image.size != (orig_w, orig_h):
                output_image = output_image.resize((orig_w, orig_h), RESAMPLE)

            save_path = output_dir / img_path.name
            output_image.save(str(save_path))
            print(f"[OK] saved: {save_path}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {img_path.name}: {e}")
            print(
                "[HINT] 可尝试：--max-side 768 或 512，"
                "或加 --enable-cpu-offload，"
                "或确认没有别的进程占用显卡"
            )

        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

        finally:
            # 主动释放引用，避免多图连续跑时显存堆积
            try:
                del input_image
            except Exception:
                pass
            try:
                del ref_image
            except Exception:
                pass
            try:
                del output_image
            except Exception:
                pass
            try:
                del output
            except Exception:
                pass
            try:
                del kwargs
            except Exception:
                pass

            cleanup_cuda()

    print("[DONE]")


if __name__ == "__main__":
    main()