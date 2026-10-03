"""把柠檬叔那张 cos 照抠成透明背景，做成能顶掉 Lucy 的贴图。**走 GPU。**

    uv run --index-url https://pypi.tuna.tsinghua.edu.cn/simple --python 3.13 \
           --with rembg --with pillow --with "onnxruntime-gpu==1.20.1" \
           --with nvidia-cudnn-cu12 --with nvidia-cublas-cu12 python tools/cutout.py

背景是真背景（天空/草地），颜色抠法没用，所以用 rembg（U²-Net 分割）。

⚠️ 踩过的坑，一条条都留在这儿：
1. **CPU vs GPU**：默认 `onnxruntime` 是 CPU 版，1200 万像素的图能把 12 个核烧满。
   必须 `onnxruntime-gpu`。
2. **cuDNN 不在系统里**：本机有 CUDA Toolkit（v11.6/11.7/12.0），但一个 cudnn*.dll 都没有。
   所以 cuDNN 从 pip 装（nvidia-cudnn-cu12），不污染系统。
3. **大版本必须对齐**：`onnxruntime-gpu 1.30` 要的是 `cublasLt64_13.dll`（**CUDA 13**），
   而 pip 的 `nvidia-cublas-cu12` 给的是 CUDA 12 的 dll → 报
   `Error loading onnxruntime_providers_cuda.dll ... depends on cublasLt64_13.dll which is missing (Error 126)`，
   然后**静默回退 CPU**。所以这里钉 `onnxruntime-gpu==1.20.1`（对应 CUDA 12）。
4. **DLL 目录要自己挂**：pip 把 dll 放在 `site-packages/nvidia/*/bin`，不在 PATH 上；
   onnxruntime 不会自己去找（PyTorch 会）。而且 **uv 的布局是 archive 覆盖层**
   （`E:\\uv_cache\\archive-v0\\<hash>\\Lib\\site-packages`），`site.getsitepackages()` **看不到它** ——
   必须遍历 `sys.path` 才找得到。
5. **别调 `session.get_providers()`**：rembg 的 `U2netSession` 没这个方法，一调就 AttributeError 崩。
   要看 provider 就用 `onnxruntime.get_available_providers()`，外加跑起来之后用 nvidia-smi 验。
"""
import glob
import os
import sys

import sys

if len(sys.argv) < 3:
    print('用法：python %s <输入图片> <输出webp路径> [分辨率]' % os.path.basename(__file__))
    sys.exit(2)
SRC, OUT = sys.argv[1], sys.argv[2]
MAX_HEIGHT = 1400  # 皮肤那个框才 693px 高，1400 够 2x 屏了
# 模型代际（查过，别再随便换）：
#   u2net            2020  ← rembg 的默认值。我当时直接抄了默认，头发丝糊，被柠檬叔判「不咋滴」
#   u2net_human_seg  2020     跟 u2net **同龄**，只是训练集偏人像 —— 换了等于没换（我推荐错过的）
#   isnet / dis_*    2022
#   sam              2023     rembg 里还是 SAM 1；SAM 3 早出了，说明这包跟得不勤
#   birefnet-*       2024 ★   arXiv 2401.03407，2025-07 还在出 v7。人像用 birefnet-portrait
# 注意 birefnet 系列的名字是**连字符**（birefnet-portrait），写上下划线会跑不起来。
MODEL = "birefnet-portrait"


def register_nvidia_dlls() -> list:
    """
    把 pip 装的 CUDA/cuDNN DLL 目录挂进 DLL 搜索路径。

    **必须遍历 sys.path** —— uv 用的是 archive 覆盖层，包不在 getsitepackages() 报的那些目录里。
    """
    added = []
    roots = list(sys.path)
    try:
        import site

        roots += list(site.getsitepackages())
    except Exception:
        pass
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pattern in ("nvidia/*/bin", "nvidia/*/lib", "nvidia/*/bin/*"):
            for path in glob.glob(os.path.join(root, *pattern.split("/"))):
                if not os.path.isdir(path) or path in added:
                    continue
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                added.append(path)
    return added


def main() -> int:
    added = register_nvidia_dlls()
    print(f"挂了 {len(added)} 个 nvidia DLL 目录", flush=True)
    for path in added[:8]:
        print(f"  {path}", flush=True)

    import onnxruntime as ort

    providers = ort.get_available_providers()
    print(f"onnxruntime {ort.__version__} 可用 provider：{providers}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        print("✗ CUDAExecutionProvider 不可用 —— 会退到 CPU，先别跑", flush=True)
        return 2

    from PIL import Image
    from rembg import new_session, remove

    # 显式点名 CUDA，不让它自己挑（挑错了就白等）。
    # 注意：U2netSession **没有** get_providers()，所以这里不打印它 —— 跑起来后用 nvidia-smi 验。
    session = new_session(MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"session 已建（{type(session).__name__}）", flush=True)

    original = Image.open(SRC).convert("RGB")
    print(f"原图 {original.size[0]}x{original.size[1]}，开始推理", flush=True)

    cut = remove(original, session=session)
    print("抠图完成", flush=True)

    bbox = cut.getbbox()
    if bbox:
        cut = cut.crop(bbox)
        print(f"裁到主体 {cut.size[0]}x{cut.size[1]}", flush=True)

    if cut.height > MAX_HEIGHT:
        ratio = MAX_HEIGHT / cut.height
        cut = cut.resize((max(1, round(cut.width * ratio)), MAX_HEIGHT), Image.LANCZOS)
        print(f"压到 {cut.size[0]}x{cut.size[1]}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cut.save(OUT, "WEBP", quality=92, method=6)
    print(f"已保存 {OUT}  {os.path.getsize(OUT) // 1024} KB  {cut.size[0]}x{cut.size[1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
