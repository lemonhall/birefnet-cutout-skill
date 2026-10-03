"""用官方 BiRefNet_HR-matting 抠图 —— 自己驱动 onnxruntime，不过 rembg。

**为什么要绕开 rembg**：rembg 把输入**固定压到 1024×1024**
（`birefnet_general.py` 里那句 `(1024, 1024)`），发丝在那一刻就没了。官方对
高分辨率的方案不是切块，而是**换 2048×2048 训练的 HR-matting 权重**，它同时输出
**软 alpha**（不是二值），所以也不需要再上 pymatting。

    uv run --index-url https://pypi.tuna.tsinghua.edu.cn/simple --python 3.13 \
           --with pillow --with numpy --with onnxruntime-gpu \
           --with "nvidia-cudnn-cu12<9.10" --with nvidia-cublas-cu12 \
           python -u tools/cutout_hr.py [分辨率]

分辨率默认 2048（官方训练分辨率）。想再细可以往上加，显存吃得也多 —— 实测 2048 在
RTX 2070 8G 上够用。
"""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

import sys

if len(sys.argv) < 3:
    print('用法：python %s <输入图片> <输出webp路径> [分辨率]' % os.path.basename(__file__))
    sys.exit(2)
SRC, OUT = sys.argv[1], sys.argv[2]
MODEL_PATH = os.environ.get('CUTOUT_MODEL', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'BiRefNet_HR-matting-epoch_135.onnx'))
DEFAULT_RES = 2048
MAX_HEIGHT = 1400

# 官方/rembg 用的都是这套 ImageNet 归一化
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def register_nvidia_dlls() -> int:
    """把 pip 装的 CUDA/cuDNN DLL 目录挂上。必须遍历 sys.path —— uv 用 archive 覆盖层。"""
    count = 0
    for root in list(sys.path):
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            if not os.path.isdir(path):
                continue
            try:
                os.add_dll_directory(path)
            except Exception:
                pass
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            count += 1
    return count


def main() -> int:
    print(f"挂了 {register_nvidia_dlls()} 个 nvidia DLL 目录", flush=True)

    import onnxruntime as ort

    print(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()}", flush=True)

    res = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RES
    if not os.path.exists(MODEL_PATH):
        print(f"模型不在：{MODEL_PATH}", flush=True)
        return 1

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    # 实测：这台 2070（8GB）**跑不了 2048** —— 那个 ONNX 形状写死 2048，
    # 而 `/decoder/decoder_block1/dec_att/aspp_deforms.2/atrous_conv` 这一层
    # 要申请**单个 3.28GB** 的缓冲，加上模型 1GB 和其他缓冲必 OOM（试过三次，
    # 显存清到 611MB 也一样）。降分辨率不行（形状写死），分块也不行（每块还是 2048）。
    # 所以要么换 lite 模型，要么**走 CPU** —— 慢，但走得通（3.28GB 走内存）。
    if os.environ.get("CUTOUT_FORCE_CPU") == "1":
        providers = ["CPUExecutionProvider"]
        print("（按 CUTOUT_FORCE_CPU=1 强制走 CPU）", flush=True)
    # ⚠️ 这个 ONNX 是**固定 2048×2048 输入**的（形状写死在模型里，传 1536 会报
    # `Got invalid dimensions ... Expected: 2048`），所以**不能靠降分辨率省显存**，
    # 分块也不行（每块进去还是 2048）。省显存只能靠下面这几个 session 开关：
    #   · arena_extend_strategy=kSameAsRequested  按需扩，别一次占一大片
    #   · cudnn_conv_algo_search=HEURISTIC        默认 EXHAUSTIVE 会预先吃掉大量显存
    #   · enable_mem_pattern=False                关掉内存复用规划
    #   · gpu_mem_limit                           给桌面留一点（2070 只有 8G，桌面占 2.5G）
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    try:
        options.enable_cpu_mem_arena = False
    except Exception:
        pass
    options.add_session_config_entry("arena_extend_strategy", "kSameAsRequested")
    options.add_session_config_entry("cudnn_conv_algo_search", "HEURISTIC")
    options.add_session_config_entry("gpu_mem_limit", str(6 * 1024 * 1024 * 1024))
    session = ort.InferenceSession(MODEL_PATH, sess_options=options, providers=providers)
    print(f"inputs={[i.name for i in session.get_inputs()]}", flush=True)
    print(f"outputs={[(o.name, o.shape) for o in session.get_outputs()]}", flush=True)
    print(f"推理分辨率 {res}x{res}", flush=True)

    original = Image.open(SRC).convert("RGB")
    print(f"原图 {original.size[0]}x{original.size[1]}", flush=True)

    # 预处理：resize → /255 → 归一化 → NCHW → float32
    small = original.resize((res, res), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

    started = time.time()
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    elapsed = time.time() - started
    print(f"推理完成 {elapsed:.2f}s（前向）", flush=True)

    # BiRefNet 的 ONNX 会吐多个输出（多层监督）。取**最后一个**，它是最终预测图。
    chosen = None
    for out in reversed(outputs):
        if out.ndim == 4 and out.shape[1] == 1 and out.shape[2] == res:
            chosen = out
            break
    if chosen is None:
        print(f"没找到形如 (1,1,{res},{res}) 的输出；实际：{[o.shape for o in outputs]}", flush=True)
        return 2
    print(f"用了输出 {chosen.shape}（共 {len(outputs)} 个）", flush=True)

    # sigmoid → 软 alpha（这就是 HR-matting 的价值：出的是软边，不是二值）
    logits = chosen[0, 0]
    alpha_small = 1.0 / (1.0 + np.exp(-logits))
    lo, hi = float(alpha_small.min()), float(alpha_small.max())
    print(f"alpha 范围 {lo:.3f}~{hi:.3f}", flush=True)

    alpha = Image.fromarray((np.clip(alpha_small, 0, 1) * 255).astype(np.uint8), mode="L")
    alpha = alpha.resize(original.size, Image.Resampling.LANCZOS)

    result = original.convert("RGBA")
    result.putalpha(alpha)

    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)
        print(f"裁到主体 {result.size[0]}x{result.size[1]}", flush=True)

    if result.height > MAX_HEIGHT:
        ratio = MAX_HEIGHT / result.height
        result = result.resize((max(1, round(result.width * ratio)), MAX_HEIGHT), Image.LANCZOS)
        print(f"压到 {result.size[0]}x{result.size[1]}", flush=True)

    # 统计"半透明像素"占比 —— 发丝就在这一档里，这个数字能直接反映发丝保住了多少
    a = np.asarray(result.getchannel("A"), dtype=np.uint8)
    total = a.size
    print(f"全透明 {np.count_nonzero(a == 0) / total * 100:.1f}%  "
          f"半透明(1~254) {np.count_nonzero((a > 0) & (a < 255)) / total * 100:.1f}%  "
          f"不透明 254+ {np.count_nonzero(a >= 254) / total * 100:.1f}%", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    result.save(OUT, "WEBP", quality=95, method=6)
    print(f"已保存 {OUT}  {os.path.getsize(OUT) // 1024} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
