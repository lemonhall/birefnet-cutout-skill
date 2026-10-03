---
name: portrait-cutout
description: 把人物从照片里抠出来（背景移除 / 抠图 / matting / 换背景 / 做头像 / 做贴图），以及为什么发丝总是糊。讲清模型代际（u2net 2020 → birefnet 2024）、rembg 把输入固定压到 1024 这件事、官方 2048 HR-matting 与它的软 alpha、8GB 显卡跑不了 2048 的实测原因、以及用「半透明像素占比」客观衡量发丝保住多少。当柠檬叔提到「抠图 / 去背景 / 抠人物 / 抠头发 / 发丝 / 换背景 / 透明底 / matting / BiRefNet / rembg / u2net」时使用。
---

# 抠人物：方法与实测边界

这份手册来自 2026-10-04 的一次完整实战：把一张 2970×4096 的真人 cos 照抠成透明背景，
前后试了三代模型、两条流水线，最后量出了**这台机器的能力边界**。

## 一、先结论

| 你要什么 | 用什么 | 代价 |
| --- | --- | --- |
| **快**（头像、配图、不讲究发丝） | `scripts/cutout_gpu.py` —— rembg + `birefnet-portrait` | GPU 秒级 |
| **发丝**（要软边、要放大看） | `scripts/cutout_hr.py` —— 官方 `BiRefNet_HR-matting` @2048 | **CPU 95 秒/张**（8GB 卡跑不了 2048，见第五节） |
| 批量 | 别用 2048 那条 ✗ 太慢；考虑 lite 模型或升显卡 | — |

两个脚本都在 `scripts/`，用法一样：

```powershell
# 快（GPU）
uv run --python 3.13 --with rembg --with pillow --with "onnxruntime-gpu==1.22.0" `
       --with "nvidia-cudnn-cu12<9.10" --with nvidia-cublas-cu12 `
       python scripts/cutout_gpu.py  <输入图片> <输出.webp>

# 发丝（官方 2048，走 CPU）
$env:CUTOUT_FORCE_CPU='1'
uv run --python 3.13 --with pillow --with numpy --with "onnxruntime-gpu==1.22.0" `
       --with "nvidia-cudnn-cu12<9.10" --with nvidia-cublas-cu12 `
       python scripts/cutout_hr.py   <输入图片> <输出.webp>
```

## 二、模型代际（**别抄默认值**）

`rembg` **默认是 `u2net`** —— 那是 **2020** 年的模型。它有 21 个可选 session，
我第一版直接抄了默认，抠出来头发丝糊成一团。**用之前先看代际**：

| session | 年代 | 说明 |
| --- | --- | --- |
| `u2net` | **2020** | rembg 的默认值 ✗ 老 |
| `u2net_human_seg` | **2020** | **跟 u2net 同龄** ✗ 只是训练集偏人像，换了等于没换（我踩过） |
| `isnet` / `dis_general_use` | 2022 | |
| `sam` | 2023 | rembg 里还是 SAM 1；SAM 3 早出了 → 这包跟得不勤 |
| `bria_rmbg` | 2024 | |
| **`birefnet-portrait`** | **2024** ★ | BiRefNet（arXiv 2401.03407）人像特化；命名是**连字符** |
| `birefnet_general` | 2024 | 通用 |
| 官方 `BiRefNet_HR-matting` | **2025-02** ★★ | **2048×2048 训练、输出软 alpha**，官方对高分辨率的正解 |

> BiRefNet 的论文：arXiv **2401.03407**（2024-01-07 投稿，2025-07 还在出 v7）；
> 发表：CAAI Artificial Intelligence Research, 2024。

## 三、发丝为什么会糊（根因，不是"模型不行"）

**rembg 把输入固定压到 1024×1024。** 源码里写死的：

```
birefnet_general.py L35:  ... (1024, 1024)                    ← 输入尺寸
                  L47:  mask = ...mode="L"                    ← 单通道掩码
                  L48:  mask.resize(img.size, LANCZOS)        ← 再放大回原尺寸
```

一张 1200 万像素的照片压到 1024 再放大回来 —— **发丝在"压下去"那一步就没了** ✗，
后面怎么放都只是糊的块状边缘。**换模型救不了**，因为换哪个都是 1024 进去。

**官方的正解不是切块**，而是换 **`BiRefNet_HR-matting`**（2048 训练、输出**软 alpha**）。
它出的是**连续 alpha**（实测范围 0.000~1.000），不是硬边 0/255 —— 所以也不需要再上 pymatting。

> ⚠️ 切块（tiling）**对这个模型无效**：那个 ONNX 把输入形状**写死成 2048**
> （传 1536 会报 `Got invalid dimensions ... Expected: 2048`），
> 所以每块进去还是 2048，显存/算力一模一样。

## 四、怎么客观判断"发丝保住了多少"

别只看眼睛（会吵）。**量「半透明像素占比」** —— 发丝在数学上就是 alpha 介于 0 和 255 之间的像素：

```python
import numpy as np
from PIL import Image
a = np.asarray(Image.open("out.webp").getchannel("A"))
t = a.size
print("全透明 %.1f%%  半透明 %.1f%%  不透明 %.1f%%" % (
    np.count_nonzero(a == 0) / t * 100,
    np.count_nonzero((a > 0) & (a < 255)) / t * 100,
    np.count_nonzero(a == 255) / t * 100))
```

**实测对比**：

| 流水线 | 裁出主体 | alpha 范围 | 半透明档 |
| --- | --- | --- | --- |
| u2net（1024） | 863 宽 ✗ 边角丢了 | 近乎硬边 | ≈ 0 ✗ |
| birefnet-portrait（rembg 1024） | 1235 宽 | 偏硬 | — |
| **官方 HR-matting @2048** | **2970×4096 整幅** ✓ | **0.000~1.000** ✓ | **24.0%** ✓✓ |

## 五、8GB 显卡跑不了 2048（实测，别再试）

`BiRefNet_HR-matting` 的 ONNX 在 2048 上会报：

```
Failed to allocate memory for requested buffer of size 3288334336     ← 单个 3.28 GB
node: /decoder/decoder_block1/dec_att/aspp_deforms.2/atrous_conv/Mul_7
```

**那一层要申请单个 3.28GB 的缓冲**（可变形卷积注意力）。加上模型本体 1GB 与其余缓冲，
8GB 卡装不下。**试过三次**（含把显存清到 611MB、并加了
`arena_extend_strategy=kSameAsRequested` / `cudnn_conv_algo_search=HEURISTIC` /
`gpu_mem_limit` 这些开关）—— **都一样 OOM** ✗。

**而且没有降档余地**：输入形状写死 2048 → **降分辨率不行** ✗，**分块也不行** ✗。

**所以**：要么走 **CPU**（`CUTOUT_FORCE_CPU=1`，实测 **95.39 秒**跑通，3.28GB 走内存），
要么换 **lite** 模型（要 torch 那条线），要么升显卡。

## 六、GPU 那套环境（只用 GPU 路线时才需要）

五个坑，从外到内（详细版见 skill `fast-downloads-and-gpu`）：

1. 装的是 **CPU 版** `onnxruntime` → 要 `onnxruntime-gpu`
2. 系统**没有 cuDNN** → 从 pip 装 `nvidia-cudnn-cu12`（不污染系统）
3. **大版本要对齐**：`onnxruntime-gpu 1.30` 要 CUDA 13（`cublasLt64_13.dll`）→
   用 **`1.22.0`**（对应 CUDA 12，也是最早有 cp313 轮子的）
4. **cuDNN 别太新**：`9.27` 对 Turing（RTX 2070, sm_75）会
   `CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED` + `sm_count: -1` → 钉 **`<9.10`**
5. **DLL 目录要自己挂**，而且 **uv 的 archive 覆盖层只有 `sys.path` 能看到**
   （`site.getsitepackages()` 看不到 → 「挂了 0 个目录」）

**怎么证明真在 GPU 上**（三条缺一不可）：
`onnxruntime.get_available_providers()` 有 `CUDAExecutionProvider` +
跑起来时 `nvidia-smi` 里能看到 python 进程与利用率跳升 +
输出里**没有** `Falling back to ['CPUExecutionProvider']`。

> 🔴 onnxruntime 缺 DLL 时是**静默回退**：只打一条 `RuntimeWarning` 然后照样把活干完（只是慢）。
> **不主动验，你根本不知道它跑在哪。**

## 七、下载模型（1.02GB，走加速）

```powershell
$url = 'https://ghfast.top/https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet_HR-matting-epoch_135.onnx'
curl.exe -L -C - --retry 8 -o models\BiRefNet_HR-matting-epoch_135.onnx $url
```

实测 972MB 的同类文件 **29 秒**下完。`scripts/cutout_hr.py` 默认去
`scripts/../models/` 找这个文件，也可以用环境变量 `CUTOUT_MODEL` 指别处。

## 八、环境与版本

- Python：**钉 3.13**（本机 `uv` 默认 3.14，很多包没 cp314 轮子）
- pip 源：`--index-url https://pypi.tuna.tsinghua.edu.cn/simple` + `NO_PROXY` 指给镜像域名
  （实测同一批包：走代理约 40 分钟 → 镜像 **17.29 秒**）
- Windows 上 Python 的中文输出会乱码 → `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`、`python -u`
