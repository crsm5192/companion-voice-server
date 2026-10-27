# 部署指南

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.10 ~ 3.11（3.11 实测可用） |
| PyTorch | 与你的 CUDA 版本匹配（开发环境 `2.3.1+cu121`） |
| ffmpeg | 必需，用于音频格式转换 |
| 磁盘 | 基座模型约 6 GB + 每个 LoRA 约 1.9 GB |
| 内存 | 建议 ≥ 8 GB（CPU 常驻模式约 6 GB） |
| 显存 | 合成时约 3.7 GB（可用 `--device cpu` 纯 CPU 跑，但慢） |

## 2. 准备模型

需要三样东西：

```
<模型目录>/                     ← CosyVoice3 基座
├── llm.pt / flow.pt / hift.pt
├── campplus.onnx
├── cosyvoice3.yaml
└── ...

<LoRA 目录>/<epoch_xx_whole.pt>  ← 你的音色 LoRA（可选）

<参考人声目录>/*.wav             ← 参考音频池（风格/情绪系统用）
```

训练 LoRA 请参考 CosyVoice 官方流程；只想跑通服务的话，可以先用基座模型默认音色（不配 `VOICE_LORA_JP`）。

> **注意**：CosyVoice3 的推理文本必须包含 `<|endofprompt|>` 标记，否则会输出 0.02 秒空音频。本服务已自动拼接，自定义改造时务必保留。

## 3. 配置

复制 `config.example.env` 为 `.env` 并按实际情况修改，或直接设置环境变量：

```bash
export VOICE_WORKDIR=/opt/voice
export VOICE_REPO=/opt/voice/CosyVoice
export VOICE_MODEL_DIR=/opt/voice/pretrained_models/Fun-CosyVoice3-0.5B
export VOICE_LORA_JP=/opt/voice/exp/cosyvoice3/llm_lora_vocals/epoch_29_whole.pt
export VOICE_VOCAL_DIR=/mnt/voice/vocals
export FFMPEG=/usr/bin/ffmpeg
```

## 4. 启动

```bash
# 常驻（推荐）：CPU 常驻，用时搬 GPU，空闲自动退回
python tts_server.py --port 8001 --gpu-idle-seconds 60

# 纯 CPU（没有显卡时）
python tts_server.py --port 8001 --device cpu

# 一直留 GPU（显存充足时，延迟最低）
python tts_server.py --port 8001 --device cuda
```

首次启动会加载模型（约 10~30 秒），看到 `监听 http://0.0.0.0:8001` 即就绪。

## 5. 验证

```bash
# 健康检查
curl http://127.0.0.1:8001/health

# 合成测试
curl -X POST http://127.0.0.1:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"こんにちは","voice":"jp","response_format":"wav"}' \
  --output test.wav
```

正常应得到一个几百 KB 的 wav；如果是 2 KB（0.02 秒），见下方排查。

## 6. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 合成出 0.02 秒空音频 | 文本缺 `<|endofprompt|>` 标记，或 LoRA 参数与模型不在同一设备 | 本仓库已修复（`ensure_loaded` 里载入 LoRA 后会对齐设备）；自定义改造时注意保留 |
| `ModuleNotFoundError: numpy` 等 | PYTHONPATH 被其他程序污染 | 服务启动时会清理非当前 venv 的 site-packages 路径 |
| `model does not exist` | 模型目录路径错，被当成 ModelScope 模型 ID 去下载 | 确认 `VOICE_MODEL_DIR` 指向本地真实目录 |
| 参考池构建失败 | `VOICE_DATA_DIR/<VOICE_SPEAKER>/text` 或 `wav.scp` 不存在 | 设置正确的 `VOICE_DATA_DIR` 与 `VOICE_SPEAKER` |
| 显存被占满 | 模型常驻 GPU | 用默认 `auto` 模式并设 `--gpu-idle-seconds`，空闲自动退回 CPU |

## 7. 开机自启（Windows）

用任务计划程序添加一项：

- 程序：`<venv>\Scripts\pythonw.exe`
- 参数：`tts_server.py --port 8001`
- 起始于：本项目目录

> 或用 `watch_tts_server.py` 做看门狗，服务异常退出时自动拉起。
