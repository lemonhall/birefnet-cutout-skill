# portrait-cutout

把人物从照片里抠出来。附带一份**实测出来的能力边界** —— 哪一代模型、什么分辨率、什么显卡跑得动什么。

> 起因：想把一张 2970×4096 的真人照抠成透明背景做贴图。试了三代模型、两条流水线，
> 顺手量出了「发丝到底能保住多少」这件平时只能靠眼睛吵的事。

## 30 秒上手

```powershell
# ① 快：GPU + rembg + 2024 的人像模型（秒级，日常够用）
uv run --python 3.13 --with rembg --with pillow `
       --with "onnxruntime-gpu==1.22.0" --with "nvidia-cudnn-cu12<9.10" --with nvidia-cublas-cu12 `
       python scripts/cutout_gpu.py input.jpg out.webp

# ② 发丝：官方 BiRefNet_HR-matting @2048（软 alpha，发丝在）
$env:CUTOUT_FORCE_CPU='1'      # 8GB 显卡跑不了 2048，见下
uv run --python 3.13 --with pillow --with numpy `
       --with "onnxruntime-gpu==1.22.0" --with "nvidia-cudnn-cu12<9.10" --with nvidia-cublas-cu12 `
       python scripts/cutout_hr.py  input.jpg out.webp
```

第二次起依赖都在缓存里，**秒开**。

## 三条最值钱的结论

**一、别抄 rembg 的默认值。** 它默认 `u2net` —— **2020 年**的模型。抠人物要用
**`birefnet-portrait`（2024）**。而且注意 `u2net_human_seg` 虽然名字像"人像特化"，
它跟 `u2net` **同龄**，换了等于没换。

**二、发丝糊的根因是分辨率，不是模型。** rembg 把输入**固定压到 1024×1024**
（源码里写死的），1200 万像素压下去再放大 —— 发丝在那一步就没了。官方的正解是
换 **2048 训练的 `BiRefNet_HR-matting`**，它输出**连续软 alpha**。切块没用
（那个 ONNX 输入形状写死 2048）。

**三、8GB 显卡装不下 2048。** 某一层要申请**单个 3.28GB** 的缓冲，三次实测同样 OOM，
加显存策略开关也没用。所以要么走 CPU（**95 秒/张**），要么换 lite 模型，要么升显卡。

## 怎么客观量"发丝保住了多少"

发丝在数学上就是 **alpha 介于 0 和 255 之间的像素**。量它：

| 流水线 | 半透明档 |
| --- | --- |
| u2net（1024） | ≈ 0 ✗ 硬边 |
| **官方 HR-matting（2048）** | **24.0%** ✓ 整幅 2970×4096 全留住 |

比"你觉得好不好看"可靠得多，也不会吵。

## 装到你的 agent 里

这个仓库的 `SKILL.md` 是**通用格式**，DSH 和 Codex 都能吃：

- **DSH**：宿主插件里注册一下（`ctx.skills.register({ content: readFileSync('SKILL.md') })`），
  或者直接把 `SKILL.md` 的内容粘进你的 `AGENTS.md`。
- **Codex**：把 `SKILL.md` 里「一、先结论」到「五、8GB 显卡跑不了 2048」这几节
  粘进 `AGENTS.md`；脚本按上面的命令直接用。

## 环境要点（踩过的坑）

```
Python         钉 3.13（uv 默认 3.14，很多包没 cp314 轮子）
pip 源         清华镜像 + NO_PROXY 直连（同一批包：代理 40 分钟 → 镜像 17 秒）
onnxruntime-gpu  1.22.0（对应 CUDA 12；1.30 要 CUDA 13）
cuDNN          <9.10（9.27 对 Turing 太新，会 HEURISTIC_QUERY_FAILED）
DLL 挂载        必须遍历 sys.path（uv 用 archive 覆盖层，getsitepackages 看不到）
验证           nvidia-smi 抓 python 进程 + 输出里没有 "Falling back to CPU"
                （onnxruntime 缺 DLL 时是静默回退，不验就不知道跑在哪）
模型下载        ghfast.top 加速 + curl -C -（972MB 实测 29 秒）
```

## 目录

```
SKILL.md                 方法论（给 agent 看的）
scripts/cutout_gpu.py    GPU 快路线：rembg + birefnet-portrait
scripts/cutout_hr.py     官方 2048 HR-matting（CPU 可跑通）
```

## License

MIT
