#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
常驻 TTS（语音合成）服务 —— 桌面宠物配套。

为「桌面伴侣 / Desktop Companion」等桌宠程序提供低延迟语音合成：
模型启动时只加载一次常驻内存，之后每次请求仅做推理，避免反复加载导致的卡顿。

- 默认把模型放 CPU（占内存不占显存），合成时自动搬上 GPU，
  空闲 gpu_idle_seconds 秒后自动退回 CPU（把显存还给 ComfyUI 等其它程序）。
- 接口：
    GET  /health            -> {"ok": true, "device": "cpu|cuda", "current_lang": ...}
    POST /tts               -> {"text", "lang"?: "jp", "speed"?: 1.0}  -> audio/wav
    POST /v1/audio/speech   -> {"input", "voice"?: "jp", "response_format"?: "mp3|wav"} -> audio
                               （OpenAI 兼容，便于任何支持该协议的客户端接入）
    POST /vc                -> 语音转换（把其它音频转成本音色）

用法：
    python tts_server.py --port 8001 --gpu-idle-seconds 60

所有模型/数据路径均可用环境变量覆盖（见 config.example.env 与 README）。
资源占用（实测）：内存 ≈ 6.0GB 常驻；显存 ≈ 3.7GB（仅合成时占用）。
"""
import os
import sys
import re
import json
import random
import argparse
import threading
import time
import subprocess
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── 路径配置（跨平台）────────────────────────────────────────────────────
# 全部可用环境变量覆盖，未设置时保持原有的 Windows 默认值不变
# （Windows 上行为与改造前逐字一致；Linux 上由部署脚本注入 VOICE_* 变量）。
#
#   VOICE_WORKDIR     工作目录（默认 Windows: C:\\voice / Linux: /opt/voice）
#   VOICE_REPO        CosyVoice 仓库目录
#   VOICE_MODEL_DIR   基座模型目录
#   VOICE_LORA_JP     日语 LoRA 权重
#   VOICE_VOCAL_DIR   纯人声参考池（风格系统依赖，245 条）
#   VOICE_REF_AUDIO   兜底参考音频
#   VOICE_DATA_DIR    训练数据目录（text / wav.scp）
#   FFMPEG            ffmpeg 可执行文件
_IS_WIN = os.name == "nt"
_DEF_WORKDIR = os.environ.get("VOICE_WORKDIR") or ("C:\\voice" if _IS_WIN else "/opt/voice")
_DEF_VOCAL = os.environ.get("VOICE_VOCAL_DIR") or (
    "C:\\voice\\vocals" if _IS_WIN else "/mnt/voice/vocals")
_DEF_REF_AUDIO = os.environ.get("VOICE_REF_AUDIO") or os.path.join(
    _DEF_VOCAL, "ref_0001.wav")
_DEF_FFMPEG = os.environ.get("FFMPEG") or ("ffmpeg.exe" if _IS_WIN else "ffmpeg")

# 训练数据里的说话人子目录名（data/<speaker>/text、wav.scp）
_SPEAKER = os.environ.get("VOICE_SPEAKER") or "speaker"

WORKDIR = os.environ.get("VOICE_WORKDIR") or _DEF_WORKDIR
REPO = os.environ.get("VOICE_REPO") or os.path.join(WORKDIR, "CosyVoice")
MODEL = os.environ.get("VOICE_MODEL_DIR") or os.path.join(
    WORKDIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
VOCAL_DIR = os.environ.get("VOICE_VOCAL_DIR") or _DEF_VOCAL
DATA_DIR = os.environ.get("VOICE_DATA_DIR") or os.path.join(WORKDIR, "data")
# ── LoRA 权重 ─────────────────────────────────────────────────────────────
# 已移除中文（zh）路径：服务只输出日语。
# 原 zh 权重 exp/cosyvoice3/llm_lora_mixed3/epoch_29_whole.pt（1.89GB）不再加载，
# 常驻内存因此减少约 1.9GB。
#
# 注意 _LORA_LAZY：LoRA 权重**按需加载**而非启动时全部读入。
# 原先 ensure_loaded() 会遍历 LORA 把所有 checkpoint 一次性载入 self.lora_sd，
# 每份约 1.89GB 常驻 RAM。现在只在真正用到该语言时才读盘（首次多几秒）。
LORA = {
    "jp": (os.environ.get("VOICE_LORA_JP") or
           os.path.join(WORKDIR, "exp", "cosyvoice3", "llm_lora_vocals",
                        "epoch_29_whole.pt")),
}
_LORA_LAZY = True
REF_AUDIO = os.environ.get("VOICE_REF_AUDIO") or _DEF_REF_AUDIO
REF_TEXT = "あんまり偉そうなこと言えないんだけどね"
FFMPEG = os.environ.get("FFMPEG") or _DEF_FFMPEG

# 清理外部注入的 PYTHONPATH 污染（某些宿主程序会把自己的 venv 塞进环境变量）。
# 仅 pop 环境变量不够：Python 启动时已把污染路径写入 sys.path，
# 必须在 import torch/transformers 前一并剔除。
# 注意：必须保留【当前 venv】自己的 site-packages，否则本服务的依赖会全部失效。
os.environ.pop("PYTHONPATH", None)
_self_prefix = os.path.normcase(os.path.realpath(sys.prefix))
sys.path[:] = [
    p for p in sys.path
    if "site-packages" not in os.path.normcase(p)
    or _self_prefix in os.path.normcase(os.path.realpath(p or "."))
]
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
import torchaudio  # noqa: E402

# 单段最大长度：CosyVoice3 的 LLM 对超长输入会提前终止（输出近静音），
# 长文本按句切段、逐段合成再拼接。120 字以内单段稳定。
MAX_SEG_LEN = 120
_SENT_END = "。！？!?．…"
# 次级断点：句末标点缺失时，中文按逗号/顿号/分号/冒号切；日语按、和助词边界切。
_ZH_SECONDARY = "，、；：,;"
_JP_PARTICLE = "はがをにでとのもねよやかなて"


def _split_text(text: str, lang: str = "jp"):
    """按行、句末标点、次级断点把文本切成 <= MAX_SEG_LEN 的段。"""
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= MAX_SEG_LEN:
            segments.append(line)
            continue
        # 长行：优先句末标点（>=20 字即切）；无则窗口内找次级断点；再不行硬切。
        buf = ""
        for i, ch in enumerate(line):
            buf += ch
            if ch in _SENT_END and len(buf) >= 20:
                segments.append(buf)
                buf = ""
                continue
            if len(buf) >= MAX_SEG_LEN:
                # 在窗口内找最后一个次级断点
                cut = -1
                for j in range(len(buf) - 1, -1, -1):
                    c = buf[j]
                    if lang == "jp":
                        if c in "、":
                            cut = j + 1
                            break
                        # 助词边界：当前字符是助词、下一字符是空格/结尾
                        if c in _JP_PARTICLE and (j + 1 == len(buf) or buf[j + 1] == " "):
                            cut = j + 1
                            break
                    elif c in _ZH_SECONDARY:
                        cut = j + 1
                        break
                if cut > 0:
                    segments.append(buf[:cut])
                    buf = buf[cut:]
                else:
                    segments.append(buf)
                    buf = ""
        if buf:
            # 剩余超长则硬切
            while len(buf) > MAX_SEG_LEN:
                segments.append(buf[:MAX_SEG_LEN])
                buf = buf[MAX_SEG_LEN:]
            if buf:
                segments.append(buf)
    return segments

# 语言检测：仅用于**拒绝中文**，不用于语言路由（服务只输出日语）。
#
# ⚠ 这是**明确的产品决策**（2026-09-21 与用户确认，选 A 而非 B），请勿改成静默处理：
#   A. 明确拒绝（当前）—— 返回 400 并说明原因。中文会被 ja_to_katakana 按汉字
#      日语读音强行转写，静默产出无意义音频；宁可让不该发的请求明确失败，
#      也不要让客户端播出垃圾音频（失败可立刻定位到"谁发了中文"）。
#   B. 中文自动转写为假名（已否决）—— 能出声，但那是"日语腔的中文读音"，
#      语义怪异，且掩盖了 Hermes 违规发送中文这一事实。
#
# 背景：Hermes 的 SOUL.md 要求语音一律用日语，但其历史记录里有 6 次发了中文
#（同一句"知道啦！以后我会时不时发语音给你…"，2026-08-24 的旧记录）。
# 这 6 次在改动后会得到 400 —— 这是**预期行为**，不是缺陷。
#
# 判定策略（两级，兼顾精度与召回）：
#   A. 含**简体中文特征字** —— 这些字在标准日语汉字中不使用
#      （如 说/説、听/聞、见/見、边/辺、为/為、时/時、间/間），精度很高；
#   B. 无假名且汉字数 >= 8 —— 兜底捕捉不含特征字的简体/繁体中文。
# 只看"无假名 + 少量汉字"会误伤全日文短语（如「電車遅延」只有 4 汉字），
# 因此 A 为主、B 为辅，并对 B 设较高阈值。
_ZH_ONLY_CHARS = set(
    "说听见边缓别们吗这么为时间让从还开关门头东车马鸟长电话语请谢对错习觉"
    "样点热欢双岁写词贝刘"          # 更多简体特征字
)
_CJK_IDEOGRAPH = re.compile(r"[\u4e00-\u9fff]")
_KANA_ANY = re.compile(r"[\u3040-\u30ff]")


class UnsupportedLanguage(ValueError):
    """请求的语言不受支持（本服务仅日语）。HTTP 层据此返回 400。"""


def _reject_if_not_japanese(text):
    """检测中文并抛出 UnsupportedLanguage；日语（含假名）放行。"""
    # 含假名 -> 基本可判为日语（中文不使用假名）
    if _KANA_ANY.search(text):
        return
    # A. 简体特征字（高精度）
    hit = _ZH_ONLY_CHARS.intersection(text)
    if hit:
        raise UnsupportedLanguage(
            f"本服务仅支持日语（中文支持已移除）。"
            f"检测到简体中文特征字 {''.join(sorted(hit))}，已拒绝。"
            f"支持的语言: jp")
    # B. 无假名且汉字较多（兜底，含繁体中文）
    n_cjk = len(_CJK_IDEOGRAPH.findall(text))
    if n_cjk >= 8:
        raise UnsupportedLanguage(
            f"本服务仅支持日语（中文支持已移除）。"
            f"文本无假名且含 {n_cjk} 个汉字，判为中文，已拒绝。"
            f"若是纯汉字日语短语，请加入假名后重试。支持的语言: jp")


GPU_IDLE_SECONDS = 60

# ── 语气控制（CosyVoice3 instruct2：自然语言情绪指令）────────────────────
# 指令词表经实测筛选（详见 ASMR_诊断报告.md）：
#   - sasayaku / soft      稳定，时长 0.83~1.07x，适合作默认
#   - amai / koe_chiisaku  有失控拉长风险（实测最高 1.94x、CV 22%），
#                          由 _synth_seg_guarded 的时长护栏兜底
# 注意：instruct2 改变的是**语速与停顿节奏**，无法真正把发声方式变成气声耳语
#      （声学实测：14 组指令仅 1 组满足耳语特征）。真耳语请走 RVC 管线。
_STYLE_INSTRUCT_JP = {
    # 耳语/气声倾向（尽力而为，非真耳语）
    "sasayaku": "囁くように話してください。",
    "mimimoto": "耳元で囁くように、息混じりで、ゆっくりと話してください。",
    "koe_chiisaku": "小さな声で、息をたくさん混ぜて、ささやいてください。",
    # 情绪
    "soft": "優しく、柔らかい感じで話してください。",
    "amai": "甘く、柔らかく、囁くように話してください。",
    "excited": "元気で明るく、勢いのある感じで話してください。",
    "sad": "少し落ち込んだ、静かな気持ちで話してください。",
    "angry": "少し怒ったように、強い調子で話してください。",
    # ASMR 场景
    "nemuri": "眠くなるような、とてもゆっくりとした静かな声で話してください。",
    "azarashi": "恥ずかしそうに、照れながら小さな声で話してください。",
}

# instruct2 的分隔标记：官方示例即此形状（CosyVoice/example.py:57/89）：
#     inference_instruct2('收到好友寄来的生日礼物…', '用四川话说这句话<|endofprompt|>', prompt)
# **该标记必需**：实测指令侧缺它会直接产出 0.02s 空音频。
_ASSIST_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_EOP = "<|endofprompt|>"


def _instruct_text(style):
    """构造 instruct2 的 instruct_text（确保带 <|endofprompt|>）。"""
    txt = _STYLE_INSTRUCT_JP[style]
    if _EOP not in txt:
        txt = txt.rstrip("。") + "。" + _EOP
    return txt


# ── 引擎选择：instruct2 并非对所有风格都可靠（ASR 实测，2026-09-21）──
# 用 whisper 对每个风格跑 5 seed、比对内容相似度（test_guard_content.py）：
#     可靠: neutral 0.989 / soft 0.938 / sasayaku 1.000 / azarashi 0.966 / nemuri 1.000
#     不可靠: amai 0.612（0/5 达标）/ excited 0.566（0/5 达标）
# 不可靠的具体表现（diagnose_v2.py 归因）：
#     amai    —— instruct2 输出天然 norm_cps < 8（护栏全越界、6/6 重试耗尽）
#     excited —— 即使 norm_cps 落在阈值内，内容仍漂移（阈值内相似度 0.26）
# 两者都属"该 instruct_text 对本 LoRA 不可用"，继续调措辞无效
#（tune_excited_direct.py 已证：初测有效的措辞在 5 seed 下方向翻转）。
#
# 对策：改用 **cross_lingual** 引擎表达这两个情绪——该通道内容可靠
#（neutral 实测 0.989+），再用**语速**与**参考音频**承载情绪差异。
# 用户盲听确认：cross_lingual + speed 1.30 比原 instruct2 指令更像"兴奋/活泼"。
_STYLE_ENGINE = {
    "amai": "cross_lingual",
    "excited": "cross_lingual",
    # azarashi 于 2026-09-22 追加：PVE 上实测同一问题——
    # 短文本（10 字）经 instruct2 生成出 9.90/5.30/6.28/5.40/6.76/12.04 秒音频
    # （正常 2~3 秒），6 次护栏重试全部越界，最终"重试耗尽，采用最接近阈值的结果"。
    # 即 instruct2 对该 instruct_text 同样不可用，与 amai/excited 同因。
    "azarashi": "cross_lingual",
    # 未列出的风格沿用 instruct2（实测可靠）
}
_ENGINE_DEFAULT = "instruct2"
# cross_lingual 通道的语速系数（相对 neutral；该通道无风格补偿表可用）
_STYLE_CL_SPEED = {
    "excited": 1.30,
    "amai": 0.95,      # 甜软 -> 略慢于 neutral
    "azarashi": 0.95,  # 害羞/小声 -> 略慢于 neutral（与 amai 同量级）
}

# 各风格对应的参考池风格标签优先序。
# 标签由 build_ref_pool.py 实际产出：neutral / soft+slow / whisper+strong+bright / neutral_1
# 匹配规则见 _ref_candidates：先精确匹配，再按子串包含匹配（如 "soft" 命中 "soft+slow"），
# 因此这里写关键词即可，不必与簇标签逐字一致。
# 参考音频的韵律会经 flow 影响输出（cross_lingual/instruct2 下 llm_prompt_speech_token
# 被删除，但 flow_prompt_speech_token 与 prompt_speech_feat 仍保留），
# 因此让参考风格与目标风格匹配，比随机抽取明显更可控。
_STYLE_REF_PREF = {
    "sasayaku": ["whisper", "soft", "neutral"],
    "mimimoto": ["whisper", "soft", "neutral"],
    "koe_chiisaku": ["whisper", "soft", "neutral"],
    "nemuri": ["soft+slow", "soft", "neutral"],
    "soft": ["soft", "neutral"],
    "amai": ["soft", "neutral"],
    "azarashi": ["soft", "neutral"],
    "excited": ["neutral", "soft"],
    "angry": ["whisper", "neutral"],
    "sad": ["soft+slow", "soft", "neutral"],
    "neutral": ["neutral", "soft"],
}

# ── 低音高参考：为"音调略高"的表情提供确定性杠杆 ──────────────────────
# 用户反馈 nemuri"音调还是觉得略高"。实测（tune_nemuri.py，同文本同 seed）：
#     参考 F0=309Hz 的切片 -> 输出 F0 287.0Hz
#     参考 F0 最低的切片    -> 输出 F0 248.4Hz   **ΔF0 = -61Hz**（内容相似度 1.000）
# 即**参考音频的音高会显著带动输出音高**，且效果稳定、可控。
# 故为这些"需要更低音"的风格优先挑选池内 F0 中位数最低的参考。
# 注：标签匹配（_STYLE_REF_PREF）无法表达"按数值取最低"，
#     因此这里在 _ref_candidates 里额外支持 "__lowf0__" 伪标签。
_LOW_F0_STYLES = ["nemuri", "sad"]

# 耳语类风格需要**同时**满足"气声（HNR 低）+ 音高低"，故用联合筛选：
#   先在池内取气声最强的一批（HNR 最低的 ~40%），再在其中按 F0 升序取最低的一批。
# 为什么不直接取 F0 最低：实测（tune_sasayaku.py）纯按 F0 取会丢掉气声质感
#   （绝对最低 F0=203Hz 的切片 HNR 仅 -3.6dB，而气声桶可达 -8dB）；
#   且"参考 F0 -> 输出 F0"**并非严格单调**（F251 参考反而使输出 +8.9Hz）。
# 实测效果（sasayaku，相对 neutral 300.5Hz）：
#   当前 whisper 桶参考 -> ΔF0 -6.0Hz
#   气声+低F0 联合筛选   -> ΔF0 **-33.6Hz**，且输出 HNR -2.4dB（气声更强）
_BREATHY_LOW_F0_STYLES = ["sasayaku", "koe_chiisaku", "mimimoto"]

# 段间停顿（秒）：按风格给不同呼吸感，替代原先所有风格一律 0.22s 的做法。
_STYLE_PAUSE = {
    "sasayaku": 0.42,
    "mimimoto": 0.48,
    "koe_chiisaku": 0.40,
    "nemuri": 0.70,
    "soft": 0.34,
    "amai": 0.36,
    "azarashi": 0.40,
    "excited": 0.18,
    "angry": 0.20,
    "sad": 0.50,
    "neutral": 0.22,
}

# 时长护栏：拦截"失控拉长"这类**段内异常**，而非风格的固有快慢。
# 口径为**片假名转写后的字/秒**（模型真正读的输入）。
#
# 为什么需要"逐风格下限"而不是单一全局阈值：
#   各风格固有语速跨度极大（实测片假名口径 amai≈5.5、mimimoto≈12.0），
#   单一全局下限会把慢风格整体误判为失控（初版就犯了这个错，
#   amai 每次都被判越界并重试 3 次，白白浪费 3 倍算力）。
#   因此下限以**该风格自身实测中位数 × _GUARD_FLOOR_RATIO** 为准。
#
# 而个别风格存在"同一风格内随种子剧烈波动"（实测 koe_chiisaku 达 2.8 倍），
# 这正是护栏该拦的目标：同风格内的离群值。
_GUARD_MAX_CPS = 22.0
_GUARD_FLOOR_RATIO = 0.55
# 逐风格实测中位数（片假名 cps, speed=1.0, 已含风格补偿），由 calibrate_speed.py 产出。
# 未列出的风格回退到 _GUARD_MIN_CPS_FALLBACK。
# ⚠ 修改参考检索策略或语速补偿后必须重跑 calibrate_speed.py 并更新本表。
_GUARD_STYLE_MEDIAN = {
    "neutral": 7.65,
    "sasayaku": 8.70,
    "mimimoto": 6.27,
    "koe_chiisaku": 6.19,
    "soft": 8.61,
    "amai": 5.10,
    "excited": 7.74,
    "sad": 8.33,
    "angry": 7.99,
    "nemuri": 7.91,
    "azarashi": 8.46,
}
_GUARD_MIN_CPS_FALLBACK = 2.5
_GUARD_MAX_RETRY = 3

# ── 内容完整性护栏（v2，数据推导）────────────────────────────────────────
# 背景：用户盲听反馈"04_甘える 前面掺杂奇怪内容、不成句子"、"05 后面有拉长电音"。
# 用 whisper ASR 客观校验后确认：instruct2 会偶发**内容漂移**
#（念出指令文本 / 重复 / 乱语），且**必然伴随输出异常变长**。
#
# 在 72 个已标注样本（derive_guard_threshold.py）上的实测分离度：
#     好 (ASR相似度>=0.85)  时长 3.28~5.08s   归一化 cps 8.07~12.50
#     差 (ASR相似度<0.60)   时长 5.44~18.52s  归一化 cps 2.21~7.54
# **两组 cps 完全不重叠**，故用时长做廉价判据（无需每次跑 ASR）。
#
# 阈值 8.0 在该语料上：漏放坏样本 0、误拦好样本 0、放行纯净度 96%。
# 旧下限 = 风格中位数×0.55（amai 仅 2.8 cps）过低，坏样本 cps 3.2~4.9 被放行
# —— 这正是 04/05/06 问题的原因。
#
# 口径：cps 先除以 speed_eff 归一化（speed 是线性时间伸缩，cps ∝ speed），
# 这样阈值与"请求的语速/风格补偿"无关。
_GUARD_MIN_NORM_CPS = 8.0
# 上限：过高的 cps 意味着内容被压缩/截断，同样不可信
_GUARD_MAX_NORM_CPS = 22.0
# 内容护栏重试次数（比旧值多：漂移概率较高，多试几个 seed 值得）
_GUARD_MAX_RETRY_V2 = 6

# 每风格语速补偿。系数 = 实测倍率的**阻尼补偿** ratio^(-0.6)，
# 只把偏离拉回一部分（而非 1/ratio 全量补偿），以保留 ASMR 应有的慢与轻，
# 避免把所有风格抹平成同一语速。clamp 到 [0.85, 1.30]。
#
# 依据 calibrate_speed.py 实测（片假名口径，6 seed 中位数，已含下表补偿）：
#   amai 0.67x / koe_chiisaku 0.81x / mimimoto 0.82x  → 偏慢，需加速
#   sasayaku 1.14x / soft 1.13x / azarashi 1.11x      → 偏快，需轻微回收
#   excited 1.01x / nemuri 1.03x / angry 1.04x / sad 1.09x → 基本正常
#
# ⚠ 重要：本表必须**在参考分层检索逻辑改动后重新标定**。
# 初版曾按"参考随机抽取"时代的实测值填（mimimoto 记 1.15 加速），
# 但参考改按风格分层后，mimimoto 的参考从 neutral 换成 whisper，
# 实测倍率由 0.83x 变为 0.95x —— 补偿方向整个反了。
# 修改参考检索策略后请重跑 calibrate_speed.py 并更新本表。
_STYLE_SPEED = {
    "amai": 1.27,         # 实测 0.67x（波动 2.39x，全 seed 都偏慢）
    "koe_chiisaku": 1.14,  # 实测 0.81x（波动 2.81x，最大）
    "mimimoto": 1.13,     # 实测 0.82x
    "neutral": 1.0,
    "excited": 0.99,
    "angry": 0.98,
    # nemuri：用户反馈"音调略高、略快"，实测原 factor 0.98 几乎不减速
    #（时长 4.14s vs neutral 4.56s，反而更快）。改为下限 0.85 以尽量放慢；
    # 音高问题由"低 F0 参考音频"解决（见 _STYLE_REF_PREF 的 pool_lowf0 思路）。
    "nemuri": 0.85,
    "sad": 0.95,
    "soft": 0.93,
    "azarashi": 0.94,
    "sasayaku": 0.92,
}
# clamp 只作用于**风格补偿系数**，不作用于调用方传入的 speed。
# 这样用户始终保留最终控制权（原先 clamp 同时限制 speed×factor，
# 导致用户传 speed=1.5 也被压到 1.30，无法手动加速）。
_SPEED_MIN = 0.85
_SPEED_MAX = 1.30

# ── instructions → style 映射（OpenAI 兼容接口的情绪指令）─────────────
# 背景：Hermes 的 text_to_speech 工具会把 `instructions`（voice-design 文本：
# 语气/情绪/节奏/耳语）透传给后端，实测其 64 次调用**全部**携带该字段，
# 且语义随场景变化（撒娇/害羞/兴奋/平静/自信…）。
#
# CosyVoice3 不支持 OpenAI 的 voice-design 语义，故这里把自由文本**归类**到
# 我们已实测标定的 10 种 style 上（见 _STYLE_INSTRUCT_JP / _STYLE_PAUSE /
# _STYLE_SPEED），从而让 Hermes 的情绪判断真正生效。
#
# 设计要点：
#   · 关键词**加权评分**而非硬匹配 —— instructions 是自由自然语言，
#     且中/日文混用（实测日文居多，也有中文）。
#   · 每类关键词带权重：越专指的词权重越高（如「はしゃぐ」比「明るく」更能定情绪）。
#   · 得分相同或全为 0 时回退 neutral，避免错误“猜”情绪。
#   · 只用于 instructions；显式传 style 时以 style 为准（显式优先）。
_INSTR_STYLE_KEYWORDS = {
    # style: [(关键词, 权重), ...]
    # 权重依据：越"专指某情绪"的词权重越高；描述风格/方式的词权重低，
    # 避免「活泼俏皮」这类方式描述压过「撒娇」这类情绪本身。
    "amai": [
        ("甘える", 3), ("甘え", 3), ("甘い", 2), ("撒娇", 4),
        ("恋人", 2), ("優しく", 1), ("温柔", 1), ("甜美", 2),
        ("可爱", 2), ("可愛", 2),
    ],
    "azarashi": [
        ("照れ", 3), ("恥ずかし", 3), ("害羞", 3), ("羞", 2),
    ],
    "excited": [
        ("はしゃぐ", 3), ("はしゃい", 3), ("嬉し", 2), ("楽しく", 2),
        ("元気", 2), ("明るく", 1), ("兴奋", 3), ("開心", 3), ("开心", 3),
        ("興奮", 2), ("勢い", 2), ("活泼", 2), ("俏皮", 2),
    ],
    "nemuri": [
        ("眠", 3), ("おやすみ", 3), ("催眠", 3), ("寝かし", 3), ("助眠", 3),
    ],
    "sasayaku": [
        ("囁", 3), ("ささや", 3), ("耳元", 3), ("耳语", 3), ("whisper", 3),
        ("小声", 2), ("轻声", 2), ("細語", 2), ("细语", 2), ("ひそひそ", 3),
    ],
    "koe_chiisaku": [
        ("息混じり", 3), ("息を", 2), ("気息", 2), ("气声", 3), ("气音", 3),
    ],
    "soft": [
        ("落ち着いて", 3), ("落ち着", 3), ("頼もしい", 2), ("穏", 2),
        ("優しく", 2), ("自然", 2), ("平静", 3), ("稳定", 2), ("认真", 2),
        ("達成感", 1), ("自信", 2),
    ],
    "sad": [
        ("悲し", 3), ("落ち込", 3), ("しんみり", 3), ("难过", 3), ("低落", 3),
    ],
    "angry": [
        ("怒", 3), ("怒り", 3), ("強い調子", 3), ("生气", 3), ("严厉", 2),
    ],
    "mimimoto": [
        ("耳元で", 3), ("そばで", 2), ("贴身", 2), ("近くで", 2),
    ],
}


# instructions 里的**恒定样板文本**：音色身份描述，每条指令几乎都含，
# 但不携带情绪信息，反而会给 amai/excited 系统性灌水（如「可愛らしい」命中 可愛、
# 「明るく」命中 明るく），淹没真正决定情绪的「はしゃぐ/照れた」等词。
# 实测 Hermes 的 10 种指令里 8 种都以此开头，故在计分前整体剔除。
_INSTR_BOILERPLATE = [
    "明るく可愛らしい日本語の若い女性声。",
    "明るく可愛らしい日本語の若い女性声",
    "明るく可愛らしい若い女性声",
    "日本語の若い女性声",
    "可愛らしい日本語の若い女性声",
    "自然で親しげな話し方",
    "自然で親しげ、",
    "親しげで",
    "自然な話し方",
]


def _strip_boilerplate(text):
    """剔除音色身份描述等恒定样板，只留下描述情绪/语气的部分。"""
    for bp in _INSTR_BOILERPLATE:
        text = text.replace(bp.lower(), "")
    return text


def instructions_to_style(instructions):
    """把 OpenAI 风格的 voice-design 文本归类到本服务的 style。

    返回 (style, score)。score==0 表示无任何关键词命中，调用方应回退 neutral。

    实现要点：
      · **先剔样板**：音色身份描述（「明るく可愛らしい日本語の若い女性声」等）
        每条都带，不区分情绪，反而干扰判断。
      · **重叠去重**：同一 style 内按关键词长度降序匹配，已认领区间不再计分，
        否则「落ち着いて」会同时命中自身与「落ち着」而重复加分。
      · 不同 style 共用同一段文本是正常的（各按自己词表计分）。
    """
    if not instructions:
        return "neutral", 0
    text = _strip_boilerplate(str(instructions).lower())
    best, best_score = "neutral", 0
    for style, kws in _INSTR_STYLE_KEYWORDS.items():
        claimed = []
        s = 0
        for kw, w in sorted(kws, key=lambda x: -len(x[0])):
            k = kw.lower()
            if not k:
                continue
            start = 0
            while True:
                i = text.find(k, start)
                if i < 0:
                    break
                end = i + len(k)
                # 与已认领区间重叠则跳过（避免子串重复计分）
                if not any(not (end <= a or i >= b) for a, b in claimed):
                    claimed.append((i, end))
                    s += w
                start = i + 1
        if s > best_score:
            best, best_score = style, s
    return best, best_score

# ── 耳语化（whisperize v2）参数 ──────────────────────────────────────────
# ⚠⚠ 实测失败，默认关闭，不建议使用 ⚠⚠
# 用户盲听反馈：dry 干净，而 v1(随机相位)/v2(噪声声码器)/加法式气声
# **全部有严重杂音**，严重度大致随注入噪声量上升。
# 诊断确认无削波、无直流、采样率正确 —— 杂音就是注入的噪声本身。
# 概念错误：真耳语是"用噪声激励替换声带振动"，而在正常语音上叠加/重建噪声
# 只能得到嘶嘶声。三个客观指标(HNR/平坦度、env_corr、相位相干性)也都无法
# 区分其优劣，因此无法靠调参挽救。
# ⇒ 需要耳语请改用 POST /vc（语音转换，不注入噪声），见 TTSService.vc()。
_WHISPER_LIFTER = 75
_WHISPER_TILT_DB = -3.0
_WHISPER_ORIG_PHASE_MIX = 0.30
_WHISPER_F_FLOOR = 200.0
_WHISPER_ENABLED = False   # 总开关：保持 False 直到有可靠方案

# /vc 输出的去齿音总开关（默认开启，见 deess_highs 说明）
_DEESS_ENABLED = True


def _resolve_style(style: str, text: str, lang: str) -> str:
    """显式 style 直接用；auto 按标点启发式判断，不再一律 neutral。

    原先 auto 恒返回 neutral，等于默认路径完全没有语气控制；
    现改为按文本特征给一个保守的默认风格（有情绪标点才升级）。
    """
    style = (style or "auto").lower()
    if style in _STYLE_INSTRUCT_JP:
        return style
    if style != "auto":
        return "neutral"
    # auto：按标点/语气词做轻量启发式
    if text.count("！") + text.count("!") >= 2:
        return "amai" if "はぁ" in text or "…" in text else "excited"
    if text.count("…") >= 3 or text.count("..") >= 3:
        return "sasayaku"
    return "neutral"

# ── 参考音频池 ──────────────────────────────────────────────────────────────
# 从训练语料（<VOICE_DATA_DIR>/<speaker> 的转录 + wav.scp）构建干净切片池。
# 原先每次合成随机挑一条作参考；实测发现随机参考会让韵律锚点漂移、输出不可控，
# 现改为按风格标签检索（标注由 build_ref_pool.py 生成）。
REF_POOL_MIN_SEC = 2.5
REF_POOL_MAX_SEC = 12.0
REF_STYLE_JSON = os.path.join(WORKDIR, "ref_pool_styles.json")


def _load_ref_style_map():
    """读取 build_ref_pool.py 产出的风格标注 -> {文件名: 风格标签}。

    文件不存在时返回空 dict，调用方自动回退到"随机抽取"的旧行为，
    保证改造可降级、不会因缺标注而启动失败。
    """
    try:
        with open(REF_STYLE_JSON, encoding="utf-8") as f:
            data = json.load(f)
        m = {it["file"]: it["style"] for it in data.get("items", [])}
        print(f"[tts-server] 参考风格标注就绪：{len(m)} 条 "
              f"({len(data.get('styles', {}))} 种风格)", flush=True)
        return m
    except Exception as e:
        print(f"[tts-server] 未找到风格标注（回退随机参考）：{e}", flush=True)
        return {}


def _load_ref_pool():
    """返回 [(wav_path, transcript), ...]，按时长过滤。"""
    text_path = os.path.join(DATA_DIR, _SPEAKER, "text")
    scp_path = os.path.join(DATA_DIR, _SPEAKER, "wav.scp")
    try:
        texts = {}
        with open(text_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    texts[parts[0]] = parts[1]
        pool = []
        with open(scp_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                utt, path = parts
                path = path.strip().strip('"')
                # wav.scp 里可能是绝对路径，需与实际存放位置一致。
                # 在 Linux 上该路径不存在，用文件名到 VOCAL_DIR 兜底解析；
                # 找不到就跳过（调用方会回退到 vocal_pool，日语实际不用本池）。
                if not os.path.isfile(path):
                    cand = os.path.join(VOCAL_DIR, os.path.basename(path))
                    if os.path.isfile(cand):
                        path = cand
                    else:
                        continue
                transcript = texts.get(utt, "")
                if len(transcript) < 4:
                    continue
                try:
                    info = torchaudio.info(path)
                    dur = info.num_frames / info.sample_rate
                except Exception:
                    continue
                if REF_POOL_MIN_SEC <= dur <= REF_POOL_MAX_SEC:
                    pool.append((path, transcript))
        print(f"[tts-server] 参考音频池就绪：{len(pool)} 条", flush=True)
        return pool
    except Exception as e:
        print(f"[tts-server] 参考池构建失败（回退固定参考）：{e}", flush=True)
        return []

# pythonw.exe（无窗口后台）下 stdout 为 None，重定向到日志文件，避免 print 崩溃
if sys.stdout is None:
    sys.stdout = sys.stderr = open(
        os.path.join(WORKDIR, "tts_server.log"), "a", encoding="utf-8", buffering=1)


def split_wav_by_silence(path, max_sec=28.0, target_sr=24000,
                         min_sil=0.25, keep_db=40):
    """把长音频按**静音边界**切成 <= max_sec 的段，返回 [(tensor(1,N), sr), ...]。

    模块级函数（而非方法），因为批量脚本 batch_vc.py 也需要它，
    且不应依赖已加载的模型实例。

    为什么不硬切：speech_tokenizer 上限是 30s，直接按 28s 硬切会切在词中间，
    VC 重建后在接缝处出现断裂/杂音。因此优先在静音处下刀。

    做法：
      1. 用 librosa.effects.split 找出所有非静音区间
      2. 贪心合并相邻区间，直到加入下一段会超过 max_sec
      3. 仍在超长的连续段（无静音）才硬切
    """
    import librosa  # noqa: E402
    y, sr = torchaudio.load(path)
    y = y.mean(dim=0) if y.dim() > 1 else y[0]
    yn = y.numpy().astype("float32")
    if sr != target_sr:
        yn = librosa.resample(yn, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if yn.size / sr <= max_sec:
        return [(torch.from_numpy(yn).unsqueeze(0), sr)]

    try:
        spans = librosa.effects.split(yn, top_db=keep_db,
                                      frame_length=2048, hop_length=512)
    except Exception:
        spans = np.array([[0, yn.size]])
    if spans.size == 0:
        spans = np.array([[0, yn.size]])

    max_n = int(max_sec * sr)
    segs, cur_start, cur_end = [], None, None
    for a, b in spans:
        if cur_start is None:
            cur_start, cur_end = int(a), int(b)
            continue
        gap = int(a) - cur_end
        if (int(b) - cur_start) > max_n:
            segs.append((cur_start, cur_end))
            cur_start, cur_end = int(a), int(b)
        elif gap >= int(min_sil * sr) and (int(b) - cur_start) > max_n * 0.8:
            segs.append((cur_start, int(b)))
            cur_start, cur_end = None, None
        else:
            cur_end = int(b)
    if cur_start is not None:
        segs.append((cur_start, cur_end))

    final = []
    for a, b in segs:
        while b - a > max_n:
            final.append((a, a + max_n))
            a += max_n
        if b - a > int(0.3 * sr):
            final.append((a, b))
    return [(torch.from_numpy(yn[a:b].copy()).unsqueeze(0), sr) for a, b in final]


# ── 去齿音 / 高频控制 ────────────────────────────────────────────────────
# 用途：实测 /vc 输出在长素材上 >8kHz 能量约为源的 4 倍（0.31% -> 1.29%），
# 可能带来齿音偏亮/嘶声。本函数做"静态高频搁架 + 动态齿音抑制"。
#
# █ 为什么这次是安全的（与失败的 whisperize 形成对比）
# 本函数在 STFT 域**只修改幅度谱、完整保留原始相位**，再做 ISTFT。
# 这等价于一个时变 EQ，不破坏相位/瞬态结构 —— 而此前 whisperize 之所以产生
# "机器人感"，根因正是随机化相位破坏了音色信息。因此本处理不会引入同类伪影。
#
# █ 注意去齿音与"气声感"存在权衡
# 耳语/气声的能量本身也集中在高频，削减过猛会把呼吸感一并抹掉。
# 故默认值偏保守，并用 _deess_validate 实测"齿音下降 vs 气声保留"的平衡。
_DEESS_DEFAULT = dict(
    shelf_freq=6500.0,     # 高频搁架起点 Hz
    shelf_db=-1.5,         # 静态削减量 dB（默认偏保守，主要靠动态压制）
    ratio_thresh=0.45,     # 齿音频段能量占比阈值，超过才开始动态压制
    max_reduce_db=-9.0,    # 动态压制上限 dB
    lowpass_hz=11000.0,    # 低通截止；0 或 None = 关闭（仅滤 >11k 的残余嘶声）
    smooth_frames=7,       # 增益平滑窗（降低调制感/抽吸感）
)
# 实测（validate_deess.py，输入 /vc 长素材输出，真实耳语素材为参照）：
#   未处理            >8k=1.29%  5-10k=1.48%  HNR=-3.9dB
#   本默认值          >8k=0.30%  5-10k=0.72%  HNR=-3.8dB  包络相关=0.997
#   真实耳语素材       >8k=0.25%  5-10k=0.95%  HNR=-7.3dB
# 即处理后高频结构贴近真实耳语，而 HNR 几乎不变（气声未被削弱）、
# RMS 不变、包络相关 0.997（动态/音节起伏未被破坏）。
# 选保守默认而非"激进档(-4dB/9k)"：后者 >8k 仅 0.07%，已低于真实耳语，
# 会把呼吸感一起抹掉。


def deess_highs(y, sr, shelf_freq=6500.0, shelf_db=-2.5, ratio_thresh=0.45,
                max_reduce_db=-8.0, lowpass_hz=10500.0, smooth_frames=7):
    """对语音做静态高频搁架 + 动态齿音抑制，可选低通。仅改幅度谱，保留相位。"""
    import librosa  # noqa: E402
    n_fft, hop = 1024, 256
    y = np.asarray(y, dtype=np.float32)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, ph = np.abs(S), np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hi_mask = freqs >= shelf_freq

    # 1) 静态高频搁架（在 shel 起点处用 1/3 倍频程余弦过渡，避免突变）
    shelf = np.ones_like(freqs)
    lo_edge = shelf_freq / 1.26
    edge = (freqs >= lo_edge) & (freqs < shelf_freq)
    if edge.any():
        t = (np.log2(np.maximum(freqs[edge], 1e-9) / lo_edge) /
             np.log2(shelf_freq / lo_edge))
        shelf[edge] = 10 ** ((shelf_db * t) / 20.0)
    shelf[hi_mask] = 10 ** (shelf_db / 20.0)

    # 2) 动态齿音抑制：按帧看高频能量占比，超阈值则额外压制高频
    hi_e = np.sum(mag[hi_mask] ** 2, axis=0)
    full_e = np.sum(mag ** 2, axis=0) + 1e-12
    ratio = hi_e / full_e
    excess = np.clip((ratio - ratio_thresh) / max(1e-6, 1.0 - ratio_thresh), 0.0, 1.0)
    gain_db = excess * max_reduce_db
    # 增益平滑（沿时间轴），避免逐帧跳变造成抽吸感
    if smooth_frames and smooth_frames > 1:
        k = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        gain_db = np.convolve(gain_db, k, mode="same")
    frame_gain = (10 ** (gain_db / 20.0)).astype(np.float32)

    mag2 = mag * shelf[:, None]
    if hi_mask.any():
        mag2[hi_mask] = mag2[hi_mask] * frame_gain[None, :]

    y_out = librosa.istft(mag2 * np.exp(1j * ph), hop_length=hop, length=y.size)

    # 3) 可选低通（滤掉 >lowpass_hz 的残余，进一步压嘶声）
    if lowpass_hz and 0 < lowpass_hz < sr / 2:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(4, lowpass_hz / (sr / 2.0), btype="low", output="sos")
            y_out = sosfilt(sos, y_out).astype(np.float32)
        except Exception:
            pass

    peak = np.abs(y_out).max()
    if peak > 1.0:
        y_out = y_out / peak * 0.99
    return y_out.astype(np.float32)


class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scale


class TTSService:
    def __init__(self, device="auto"):
        # device: "auto"=CPU常驻/用时搬GPU/空闲回CPU；"cpu"=纯CPU；"cuda"=一直留GPU
        self.device = device
        self.lock = threading.Lock()
        self.cosyvoice = None
        self.current_lang = None
        self.on_gpu = False
        self.last_use = 0.0
        self.lora_sd = {}
        self.ref_pool = _load_ref_pool()
        self.ref_style = _load_ref_style_map()
        # 纯人声切片池（demucs 分离 + VAD 重切）：日语跨语种只需音频，优先用这批。
        self.vocal_pool = sorted(
            os.path.join(VOCAL_DIR, f)
            for f in os.listdir(VOCAL_DIR)
            if f.endswith(".wav")) if os.path.isdir(VOCAL_DIR) else []
        if self.vocal_pool:
            print(f"[tts-server] 纯人声参考池就绪：{len(self.vocal_pool)} 条", flush=True)
        # 按风格预分组，避免每次合成都遍历全池
        self._vocal_by_style = {}
        for p in self.vocal_pool:
            st = self.ref_style.get(os.path.basename(p))
            if st:
                self._vocal_by_style.setdefault(st, []).append(p)
        if self._vocal_by_style:
            print("[tts-server] 按风格分组：" + ", ".join(
                f"{s}={len(v)}" for s, v in sorted(self._vocal_by_style.items())), flush=True)
        # 按 F0 从低到高排序的切片（供 nemuri/sad 取低音高参考）。
        # F0 来自 build_ref_pool.py 的实测标注（ref_pool_styles.json 的 f0 字段）。
        self._vocal_by_f0 = []
        # 气声 + 低音高 联合筛选池（供 sasayaku 等耳语类使用）：
        # 取 HNR 最低的一批（气声最强），再在其中按 F0 升序排列。
        self._vocal_breathy_lowf0 = []
        try:
            with open(REF_STYLE_JSON, encoding="utf-8") as f:
                items = [it for it in json.load(f).get("items", [])
                         if it.get("f0") and os.path.isfile(it["path"])]
            ranked = sorted(items, key=lambda it: it["f0"])
            self._vocal_by_f0 = [it["path"] for it in ranked]
            if self._vocal_by_f0:
                print(f"[tts-server] 低音高参考就绪：最低 F0="
                      f"{ranked[0]['f0']:.0f}Hz (共 {len(self._vocal_by_f0)} 条按 F0 排序)",
                      flush=True)
            # 联合：先取气声最强的前 40%，再按 F0 升序
            n_breathy = max(20, int(len(items) * 0.4))
            breathy = sorted(items, key=lambda it: it["hnr"])[:n_breathy]
            self._vocal_breathy_lowf0 = [it["path"]
                                         for it in sorted(breathy, key=lambda it: it["f0"])]
            if self._vocal_breathy_lowf0:
                top = sorted(breathy, key=lambda it: it["f0"])[0]
                print(f"[tts-server] 气声+低音高参考就绪：最低 F0={top['f0']:.0f}Hz "
                      f"(HNR={top['hnr']:.1f}dB, 共 {len(self._vocal_breathy_lowf0)} 条)",
                      flush=True)
        except Exception as e:
            print(f"[tts-server] 低音高参考不可用（跳过）：{e}", flush=True)

    def _ref_candidates(self, style):
        """按风格偏好返回候选参考，**分层**返回：[(层级, [路径…]), …]。

        分层而非简单拼接很重要：neutral 桶最大（120/245），若把所有偏好拼成
        一个长列表再用 seed 取模，索引几乎必然落进 neutral 段，风格匹配会被架空。
        因此按偏好优先级分层，由 pick_ref 只在最高非空层内轮换。

        特殊：若 style 在 _LOW_F0_STYLES 中（如 nemuri/sad），**最优先**返回
        池内 F0 中位数最低的一层 —— 实测参考音高会显著带动输出音高
        （ΔF0 达 -61Hz，见 _LOW_F0_STYLES 说明）。
        """
        if not self._vocal_by_style:
            return [(0, list(self.vocal_pool))]
        tiers = []
        # 低音高优选层（仅对需要的风格）
        if style in _LOW_F0_STYLES and self._vocal_by_f0:
            n = max(8, len(self._vocal_by_f0) // 20)   # 取最低的约 5%（至少 8 条）
            tiers.append((0, list(self._vocal_by_f0[:n])))
        # 气声+低音高联合优选层（耳语类）：既保气声又压音高
        if style in _BREATHY_LOW_F0_STYLES and self._vocal_breathy_lowf0:
            n = max(10, len(self._vocal_breathy_lowf0) // 10)
            tiers.append((0, list(self._vocal_breathy_lowf0[:n])))
        prefs = _STYLE_REF_PREF.get(style) or _STYLE_REF_PREF["neutral"]
        for want in prefs:
            got = []
            # 1) 精确命中
            if want in self._vocal_by_style:
                got.extend(self._vocal_by_style[want])
            # 2) 子串命中（如 "soft" 命中 "soft+slow"）
            for lbl, paths in self._vocal_by_style.items():
                if lbl != want and want in lbl:
                    got.extend(paths)
            seen, uniq = set(), []
            for p in got:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            if uniq:
                tiers.append((len(tiers), uniq))
        return tiers or [(0, list(self.vocal_pool))]

    def pick_ref(self, lang, style="neutral", pick=0):
        """按语言与风格挑参考。

        只在**最高优先级的非空层**内做确定性轮换，保证参考风格真正匹配目标风格；
        不再使用 random.choice（会让韵律锚点漂移），也不再跨层取模。
        """
        if lang == "jp" and self.vocal_pool:
            tiers = self._ref_candidates(style)
            if tiers:
                best_paths = tiers[0][1]
                return best_paths[pick % len(best_paths)], ""
        if self.ref_pool:
            return self.ref_pool[pick % len(self.ref_pool)]
        return REF_AUDIO, REF_TEXT

    def ensure_loaded(self, lang):
        if lang not in LORA:
            raise ValueError(f"不支持的语言: {lang}（本服务仅支持日语 jp）")
        if self.cosyvoice is None:
            print("[tts-server] 首次加载 CosyVoice3（约 10~30 秒）…", flush=True)
            self.cosyvoice = AutoModel(model_dir=MODEL)
            qm = self.cosyvoice.model.llm.llm.model.model
            for layer in qm.layers:
                attn = layer.self_attn
                for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    setattr(attn, n, LoRALinear(getattr(attn, n), 16, 32))
            # 新建的 LoRA 参数默认在 CPU，需与主干保持一致（见下方 load 后的同款处理）
            try:
                dev = next(self.cosyvoice.model.llm.parameters()).device
                self.cosyvoice.model.llm.to(dev)
            except StopIteration:
                pass
            self.cosyvoice.model.llm.eval()
            self.lora_sd = {}
            if not _LORA_LAZY:
                # 旧行为：启动时全部读入（每份约 1.89GB 常驻）
                for l in LORA:
                    sd = torch.load(LORA[l], map_location="cpu")
                    self.lora_sd[l] = {k: v for k, v in sd.items()
                                       if k not in ("epoch", "step")}
            print("[tts-server] 模型就绪（CPU 常驻，LoRA 按需加载）", flush=True)
        # 语言切换：同一份权重已加载则跳过；否则按需读盘
        if self.current_lang != lang:
            if lang not in self.lora_sd:
                t0 = time.time()
                sd = torch.load(LORA[lang], map_location="cpu")
                self.lora_sd[lang] = {k: v for k, v in sd.items()
                                      if k not in ("epoch", "step")}
                print(f"[tts-server] LoRA[{lang}] 已载入 "
                      f"({time.time()-t0:.1f}s)", flush=True)
            self.cosyvoice.model.llm.load_state_dict(self.lora_sd[lang], strict=True)
            # 关键：LoRALinear 是在包装时新建的模块，其 lora_A/lora_B 参数默认在 CPU。
            # 若此时主干已在 GPU，就会触发 cuda:0 与 cpu 混算——PyTorch 会静默降级，
            # 最终产出 0.02s 空音频（质检护栏会一直重试但无法修复）。
            # 因此载入权重后必须把整个 LLM 再对齐一次设备。
            try:
                dev = next(self.cosyvoice.model.llm.parameters()).device
                self.cosyvoice.model.llm.to(dev)
            except StopIteration:
                pass
            self.current_lang = lang

    def to_gpu(self):
        if self.device == "cpu" or self.on_gpu:
            return
        # 分别搬三个子模型（llm + flow + hift，cosyvoice.model 本身不是
        # nn.Module）：hifigan(self.hift) 若留在 CPU，低语速(speed<1)时
        # 最后一块 mel 太短（< kernel 4）会在 f0_predictor 崩溃。
        self.cosyvoice.model.llm.to("cuda")
        self.cosyvoice.model.flow.to("cuda")
        self.cosyvoice.model.hift.to("cuda")
        self.on_gpu = True
        print("[tts-server] 已搬上 GPU", flush=True)

    def to_cpu(self):
        if self.on_gpu:
            self.cosyvoice.model.llm.to("cpu")
            self.cosyvoice.model.flow.to("cpu")
            self.cosyvoice.model.hift.to("cpu")
            self.on_gpu = False
            print("[tts-server] 已退回 CPU（显存还给 ComfyUI）", flush=True)

    def idle_tick(self):
        with self.lock:
            if (self.device == "auto" and self.on_gpu and GPU_IDLE_SECONDS > 0
                    and (time.time() - self.last_use) > GPU_IDLE_SECONDS):
                self.to_cpu()

    def _set_seed(self, seed):
        """固定所有随机源。

        实测：不固定种子时，同文本+同参考+同指令的输出时长 CV 高达 10.9%~22.3%
        （同一指令下最大差 1.8 倍）；固定后 CV = 0.0%。这是"怎么调都不对"的根因，
        因此每次合成都必须可复现——seed 会写入 diag 日志，便于日后原样复现。
        """
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as _np
            _np.random.seed(seed % (2 ** 32))
        except Exception:
            pass

    def _engine_of(self, style_resolved):
        """该风格使用哪个引擎（instruct2 / cross_lingual）。"""
        if style_resolved == "neutral":
            return "cross_lingual"
        return _STYLE_ENGINE.get(style_resolved, _ENGINE_DEFAULT)

    def _synth_seg(self, text_tts, style_resolved, ref_audio, speed):
        """按风格合成单段（引擎由 _STYLE_ENGINE 决定）。

        · cross_lingual：内容可靠（实测 neutral 0.989+、amai/excited 换用后
          不再漂移），但**不带情绪指令** —— 情绪由语速与参考音频承载。
          沿用带 assistant 前缀的既有可用形式。
        · instruct2：按**官方约定**把 `<|endofprompt|>` 放在**指令侧**
          （CosyVoice/example.py:57/89 即此形状）：
              inference_instruct2('收到好友寄来的生日礼物…',      <- tts_text 纯文本
                                  '用四川话说这句话<|endofprompt|>', <- 指令带标记
                                  prompt_wav)
          实测（72 样本 ASR 校验）：
            A 前缀在正文侧、指令无标记 -> 平均相似度 0.645（最差）
            B 正文纯、指令带 assistant+标记 -> 0.737
            C 正文纯、指令仅加标记        -> 0.782（最佳）
          且**指令侧缺 `<|endofprompt|>` 会直接产出空音频**（0.02s），故该标记必需。
        """
        if self._engine_of(style_resolved) == "cross_lingual":
            return [j["tts_speech"] for j in self.cosyvoice.inference_cross_lingual(
                _ASSIST_PREFIX + text_tts, ref_audio, stream=False, speed=speed)]
        return [j["tts_speech"] for j in self.cosyvoice.inference_instruct2(
            text_tts, _instruct_text(style_resolved),
            ref_audio, stream=False, speed=speed)]

    def _synth_seg_guarded(self, text_tts, style_resolved, ref_audio, speed,
                           base_seed, n_chars, diag, guard_speed=None):
        """带**内容完整性**护栏的单段合成（v2）。

        为什么改：旧版只防"失控拉长"，下限用的是 `风格中位数×0.55`
        （amai 仅 2.8 cps），而内容漂移样本的 cps 约 3.2~4.9 —— 恰好被放行，
        导致用户听到"前面掺杂奇怪内容、不成句子"（已由 whisper ASR 证实）。

        新判据：**归一化 cps**（= cps / speed_eff）落在 [_GUARD_MIN_NORM_CPS,
        _GUARD_MAX_NORM_CPS] 内。该阈值由 72 个 ASR 标注样本推导，
        好/坏样本在归一化 cps 上完全不重叠（8.07~12.50 vs 2.21~7.54）。

        漂移必然表现为"输出异常变长"（模型念了指令或重复），
        故用时长的倒数（cps）即可廉价识别，无需每次跑 ASR。

        越界则换 seed 重试（_GUARD_MAX_RETRY_V2 次），
        全部越界时取最接近阈值的一次，保证不返回空音频。
        """
        sr = self.cosyvoice.sample_rate
        sp = max(speed, 1e-6)
        # 阈值分两条通道（依据不同）：
        #  · instruct2：漂移必然伴随"念出多余内容"-> 输出变长 -> 用 8.0（数据推导）
        #  · cross_lingual：语速天然更慢且方差大（neutral 实测 4.12~9.12s，
        #    但内容相似度 0.99+）。它"慢"是合法的，不是漂移，
        #    故用宽松下限，否则会把好输出误判并反复重试。
        # 注：amai/excited 已改走 cross_lingual（见 _STYLE_ENGINE），
        #     因此它们也适用宽松下限。
        if self._engine_of(style_resolved) == "cross_lingual":
            lo, hi = 2.0, _GUARD_MAX_NORM_CPS
        else:
            lo, hi = _GUARD_MIN_NORM_CPS, _GUARD_MAX_NORM_CPS
        best = None
        best_err = None
        for attempt in range(_GUARD_MAX_RETRY_V2):
            self._set_seed(base_seed + attempt)
            chunks = [c for c in self._synth_seg(text_tts, style_resolved, ref_audio, speed)
                      if c is not None and c.shape[1] > 0]
            if not chunks:
                continue
            n_frames = sum(c.shape[1] for c in chunks)
            dur = n_frames / sr
            cps = n_chars / dur if dur > 0 else 0.0
            norm_cps = cps / sp            # 归一化：消除 speed 的线性影响
            if lo <= norm_cps <= hi:
                if attempt > 0:
                    diag.write(f"[synth]   guard: attempt{attempt} "
                               f"norm_cps={norm_cps:.1f} 通过\n")
                return chunks
            err = (lo - norm_cps) if norm_cps < lo else (norm_cps - hi)
            # 仅 instruct2 通道的"过慢"才提示内容漂移；cross_lingual 天然更慢属正常
            drift_hint = (" <-疑似内容漂移"
                          if (self._engine_of(style_resolved) == "instruct2"
                              and norm_cps < lo) else "")
            diag.write(f"[synth]   guard: attempt{attempt} dur={dur:.2f}s "
                       f"cps={cps:.1f} norm_cps={norm_cps:.1f} "
                       f"越界(允许 {lo:.1f}~{hi:.1f}, speed={sp:.2f}){drift_hint}\n")
            if best_err is None or err < best_err:
                best, best_err = chunks, err
        diag.write("[synth]   guard: 重试耗尽，采用最接近阈值的结果"
                   "（可能含内容漂移，请留意）\n")
        return best or []

    def _apply_whisperize(self, audio, mix, lifter, diag, seed,
                          tilt_db=_WHISPER_TILT_DB,
                          orig_phase_mix=_WHISPER_ORIG_PHASE_MIX):
        """对合成结果做耳语化（气声）后处理 —— v2 噪声声码器。

        █ v1 的失败：听起来像"赛博坦星人"
        v1 用「每帧独立随机相位 + 轻度幅度谱平滑」。两个原因导致机械感：
          1. 随机相位丢弃了原始相位结构，而相位携带大量瞬态/音色信息；
          2. 平滑不足时谐波尖峰仍在，与随机相位叠加产生相位梳状效应。

        █ v2 的正确做法
        真耳语 = 共振峰（声道形状）不变、激励源由周期脉冲换成湍流噪声。故：
          1. 倒谱包络提取共振峰，lifter 取在 F0 倒频之下（≈89 @270Hz）以去净谐波
          2. 激励用**倾斜噪声**（-3dB/oct 近似声门气流），而非白噪声（过亮→电子味）
          3. 相位主要用**噪声自身相位**（连续信号、帧间相干，不再每帧随机）
          4. 融合 **30% 原始相位**把音色/瞬态找回来  ← 消除机械感的关键
          5. 低频增益**有界钳制**（v1 的 (f/500)^slope 在 f→0 时发散，
             会把输出变成低频嗡鸣，实测质心塌到 11Hz）

        █ 验证（tune_whisper_v2.py，以真实耳语切片为客观目标）
          正向 干声→耳语：HNR -9.3dB、平坦 0.066、质心 2775Hz、env_corr 0.965
          反向 真耳语→再处理：质心 3312→3181Hz 几乎不变、env_corr 0.954
          对比 L60_tilt0：平坦 0.243（过噪）、质心 4629Hz（过亮）
          对比 L60_纯噪声相位：质心 1932Hz（闷）、env_corr 0.890

        librosa 采用**惰性导入**：仅此可选路径需要，缺失时只跳过耳语化。
        """
        import librosa  # noqa: E402  惰性：仅耳语化路径需要
        sr = self.cosyvoice.sample_rate
        y = audio.squeeze(0).detach().cpu().numpy().astype("float32")

        n_fft, hop = 1024, 256
        n = y.size
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag, orig_ph = np.abs(S), np.angle(S)

        # 1) 倒谱包络（抹谐波、留共振峰）
        log_mag = np.log(mag + 1e-8)
        cep = np.fft.irfft(log_mag, axis=0)
        cep[lifter:] = 0.0
        env = np.maximum(np.exp(np.fft.rfft(cep, axis=0).real), 1e-8)

        # 2) 倾斜噪声激励（有界增益，避免低频发散）
        rng = np.random.default_rng(seed % (2 ** 32))
        w = rng.standard_normal(n).astype(np.float32)
        W = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1 / sr)
        gain = (np.maximum(f, _WHISPER_F_FLOOR) / 1000.0) ** (tilt_db / 6.0206)
        nz = np.fft.irfft(W * gain, n).astype(np.float32)
        nz = nz / (np.abs(nz).max() + 1e-9)

        N = librosa.stft(nz, n_fft=n_fft, hop_length=hop)
        nmag = np.abs(N) + 1e-8
        noise_ph = N / nmag

        # 3)+4) 噪声相位与原始相位按比例融合（单位幅值）
        if orig_phase_mix > 0:
            mixed = (1 - orig_phase_mix) * noise_ph + orig_phase_mix * np.exp(1j * orig_ph)
            ph = mixed / (np.abs(mixed) + 1e-9)
        else:
            ph = noise_ph

        y_w = librosa.istft(env * (nmag / (nmag.mean() + 1e-8)) * ph,
                            hop_length=hop, length=n)

        # 响度对齐后按 mix 插值
        rms_in = np.sqrt(np.mean(y ** 2) + 1e-12)
        rms_w = np.sqrt(np.mean(y_w ** 2) + 1e-12)
        y_w = y_w * (rms_in / rms_w)
        y_out = (1.0 - mix) * y + mix * y_w
        peak = np.abs(y_out).max()
        if peak > 1.0:
            y_out = y_out / peak * 0.99

        diag.write(f"[synth] whisperize(v2): mix={mix} lifter={lifter} "
                   f"tilt={tilt_db} orig_phase={orig_phase_mix} frames={n}\n")
        return torch.from_numpy(y_out.astype(np.float32)).unsqueeze(0)

    def _split_wav_by_silence(self, path, max_sec=28.0, **kw):
        """委托给模块级 split_wav_by_silence（批量脚本也复用同一实现）。"""
        return split_wav_by_silence(
            path, max_sec=max_sec,
            target_sr=self.cosyvoice.sample_rate, **kw)

    @staticmethod
    def _rms_db(arr):
        """计算 RMS（dBFS）。用于 /vc 的能量护栏。"""
        a = np.asarray(arr, dtype=np.float64).ravel()
        if a.size == 0:
            return -120.0
        return float(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12))

    def vc(self, source_path, prompt_path=None, speed=1.0, style="neutral",
           auto_split=True, gap_sec=0.25, deess=None, seed=1234):
        """语音转换（Voice Conversion）：保留 source 的内容/韵律/气声，换成 prompt 的音色。

        █ 为什么这条路径优于自研耳语化 DSP
        用户实测：dry 干净，而所有基于"注入噪声"的耳语化（随机相位重建、噪声声码器、
        叠加气声层）都产生严重杂音，且严重度大致随注入噪声量上升。
        诊断确认无削波/无直流/采样率正确 —— 杂音就是注入的噪声本身。

        概念错误在于：真耳语是用噪声激励**替换**声带振动（整体改变发声机制），
        而在正常语音上叠加噪声只会得到嘶嘶声。

        VC 则完全不加噪声：它从 source 提取 speech token（保留内容、韵律、
        以及气声的时频结构），用 prompt 提供音色 embedding，直接交给 flow 重建。
        实测 HNR 保持 -2.5dB（source -3.3 / prompt +6.2），即气声得以保留；
        且 >8kHz 能量不升高，无嘶声。

        █ 用途
        从别的 ASMR 作品里剪出耳语人声片段作为 source，即可得到"昴音色的耳语"。
        这正好绕开"本角色只有一部作品、且人在病休"的素材困境：
        音色由 prompt 锚定，source 只提供"怎么发声"。

        █ 长音频自动处理
        speech_tokenizer 要求单段 <=30s。auto_split=True 时会按**静音边界**自动切段
        （硬切会切在词中，接缝处会有断裂），逐段转换后拼接，段间插入 gap_sec 静音。

        █ 关键发现：prompt 会把 source 的气声"拉向自己"（第三方素材验证所得）
        固定同一耳语 source，只换 prompt（test_prompt_bias.py）：
            normal prompt（正常发声）-> 输出 HNR +2.8dB
            whisper prompt（气声切片）-> 输出 HNR -1.4dB     差 4.1dB
        而说话人相似度几乎不变（0.427 vs 0.425）—— 用气声 prompt 是净收益。
        （此前用昴自己的耳语切片做 source，因 source/prompt 同源而掩盖了该效应。）

        组合测试（test_vc_improvements.py）进一步确认：气声 prompt 在 4/4 组对比中
        都给出更低的输出 HNR，且 track4 的 sim昴 由 0.735 提升到 0.822。
        因此 **prompt 省略时默认按 style 取气声参考**（见下方 prompt 选择逻辑）。

        █ 输入电平：不要归一化
        第三方素材电平极低（如 -40.5dB）时，直觉上应归一化再送 VC。
        但实测相反（test_vc_improvements.py）：
            track2 原始     -> 出 HNR -0.0dB、sim源 0.376
            track2 归一化后 -> 出 HNR +6.7dB、sim源 0.213
        放大增益同时放大了底噪，反而污染 speech token。
        **因此本函数不对输入做电平归一化**，请勿"好心"加上。
        """
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"source 不存在: {source_path}")

        with self.lock:
            self.ensure_loaded("jp")
            self.to_gpu()
            if not prompt_path:
                # 默认优先取**气声参考**：prompt 会把 source 的气声拉向自己
                # （实测差 4.1dB），而音色相似度不受影响，故这是净收益。
                # pick_ref 对 sasayaku 已有 whisper 偏好，这里对任意 style
                # 都兜底试一次 whisper，取不到再退回原 style。
                prompt_path = None
                for st_try in ([style or "neutral"] +
                               ([] if (style or "neutral") == "sasayaku" else ["sasayaku"])):
                    cand, _ = self.pick_ref("jp", st_try, pick=0)
                    rs = self.ref_style.get(os.path.basename(cand), "")
                    if "whisper" in rs:
                        prompt_path = cand
                        break
                    if prompt_path is None:
                        prompt_path = cand      # 记录首个候选作兜底
            if not os.path.isfile(prompt_path):
                raise FileNotFoundError(f"prompt 不存在: {prompt_path}")

            diag = open(os.path.join(WORKDIR, "tts_synth_diag.log"),
                        "a", encoding="utf-8")
            srcsr = self.cosyvoice.sample_rate
            try:
                diag.write(f"[vc] source={os.path.basename(source_path)} "
                           f"prompt={os.path.basename(prompt_path)} speed={speed} "
                           f"auto_split={auto_split}\n")

                # prompt 同样受 30s 限制；超长时取前 25s
                pinfo = torchaudio.info(prompt_path)
                if pinfo.num_frames / pinfo.sample_rate > 29.5:
                    py, psr = torchaudio.load(prompt_path)
                    py = py.mean(dim=0) if py.dim() > 1 else py[0]
                    py = py[: int(25.0 * psr)]
                    fd, ptmp = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    torchaudio.save(ptmp, py.unsqueeze(0), psr)
                    prompt_use = ptmp
                    diag.write(f"[vc] prompt 超长，截取前 25s\n")
                else:
                    prompt_use, ptmp = prompt_path, None

                try:
                    if auto_split:
                        segs = self._split_wav_by_silence(source_path)
                    else:
                        info = torchaudio.info(source_path)
                        if info.num_frames / info.sample_rate > 29.5:
                            raise ValueError(
                                f"source 时长 "
                                f"{info.num_frames/info.sample_rate:.1f}s 超过 30s，"
                                "请设 auto_split=true 或先切分")
                        y0, s0 = torchaudio.load(source_path)
                        y0 = y0.mean(dim=0) if y0.dim() > 1 else y0[0]
                        segs = [(y0.unsqueeze(0), s0)]

                    diag.write(f"[vc] 切分为 {len(segs)} 段\n")
                    outs = []
                    for i, (seg, ssr) in enumerate(segs):
                        fd, tmp = tempfile.mkstemp(suffix=".wav")
                        os.close(fd)
                        try:
                            torchaudio.save(tmp, seg, ssr)
                            # ── 能量护栏 ──────────────────────────────────
                            # 实测：336 段批量成品中有 11 段近静音（RMS -45~-61dB），
                            # 而对应源段正常（-23dB）。
                            # 进一步验证（segments 34-37 同源对比）：
                            #     segs[34] -23.5 -> -31.6dB  (正常)
                            #     segs[35] -23.1 -> -56.7dB  (失败, -33.7dB)
                            #     segs[36] -22.6 -> -45.7dB  (失败)
                            #     segs[37] -23.0 -> -33.8dB  (正常)
                            # 且 **attempt0/1/2 输出逐位相同** —— inference_vc
                            # 不响应 torch.manual_seed，故"换种子重试"无效，本实现不做无谓重试。
                            # 正确策略：能量不达标即**回退使用源音频本身**
                            #（保留原始演出与内容，只是该段未换音色），
                            # 宁可个别段不换音色，也绝不产出近静音段。
                            src_rms = self._rms_db(seg.squeeze(0).numpy())
                            self._set_seed(seed + i)
                            ch = [j["tts_speech"] for j in self.cosyvoice.inference_vc(
                                tmp, prompt_use, stream=False, speed=speed)]
                            ch = [c for c in ch if c is not None and c.shape[1] > 0]
                            cand = torch.cat(ch, dim=1) if ch else None
                            if cand is not None:
                                out_rms = self._rms_db(cand.squeeze(0).numpy())
                                if out_rms > src_rms - 12.0:
                                    outs.append(cand)
                                    diag.write(f"[vc]   段{i+1}: in={seg.shape[1]} "
                                               f"out={cand.shape[1]} "
                                               f"src={src_rms:.1f}dB out={out_rms:.1f}dB\n")
                                else:
                                    # 回退：用源音频（不换音色，但保证有声）
                                    outs.append(seg)
                                    diag.write(f"[vc]   段{i+1}: 能量异常"
                                               f"(src={src_rms:.1f}dB out={out_rms:.1f}dB)"
                                               f" -> 回退源音频\n")
                            else:
                                outs.append(seg)
                                diag.write(f"[vc]   段{i+1}: 输出为空 -> 回退源音频\n")
                        finally:
                            try:
                                os.remove(tmp)
                            except OSError:
                                pass

                    if not outs:
                        audio = torch.zeros(1, 100)
                    else:
                        gap = torch.zeros(1, int(gap_sec * srcsr), dtype=outs[0].dtype)
                        merged = []
                        for i, o in enumerate(outs):
                            if i:
                                merged.append(gap)
                            merged.append(o)
                        audio = torch.cat(merged, dim=1)
                    diag.write(f"[vc] 总帧数={audio.shape[1]}\n")

                    # 去齿音 / 高频控制（实测长素材 >8k 能量约为源 4 倍）。
                    # 只改幅度谱、保留相位，因此不会引入 whisperize 那类机械感。
                    # 但去齿音会与"气声感"抢高频能量，故默认保守，可显式关闭。
                    if deess is None:
                        deess = _DEESS_ENABLED
                    if deess:
                        yv = audio.squeeze(0).numpy().astype(np.float32)
                        yv = deess_highs(yv, self.cosyvoice.sample_rate,
                                         **_DEESS_DEFAULT)
                        audio = torch.from_numpy(yv).unsqueeze(0)
                        diag.write("[vc] deess 已应用 "
                                   + " ".join(f"{k}={v}" for k, v in _DEESS_DEFAULT.items())
                                   + "\n")
                finally:
                    if ptmp:
                        try:
                            os.remove(ptmp)
                        except OSError:
                            pass
            finally:
                diag.close()
                self.last_use = time.time()

        fd, wavpath = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            torchaudio.save(wavpath, audio.cpu(), self.cosyvoice.sample_rate)
            with open(wavpath, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(wavpath)
            except OSError:
                pass

    def synth(self, text, lang="auto", speed=1.0, style="auto", seed=None,
              whisper_mix=0.0, whisper_lifter=_WHISPER_LIFTER):
        # 本服务只输出日语：lang 一律归一到 jp（保留参数以兼容既有调用方）
        lang = "jp"
        # 明确拒绝中文，避免被当作日语强行转写后产出垃圾音频
        _reject_if_not_japanese(text)
        # seed=None 时随机取一个并**写入日志**，保证事后仍可原样复现。
        if seed is None:
            seed = random.randrange(1, 2 ** 31 - 1)
        diag = open(os.path.join(WORKDIR, "tts_synth_diag.log"), "a", encoding="utf-8")
        diag.write(f"[synth] lang={lang} speed={speed} style={style} seed={seed} "
                   f"text_len={len(text)} text_head={text[:80]!r}\n")
        with self.lock:
            self.ensure_loaded(lang)
            self.to_gpu()
            try:
                # 语气：auto 按标点启发式；显式风格直接采用。
                style_resolved = _resolve_style(style, text, lang)
                # 语速补偿：按实测倍率修正风格固有语速（如 amai 系统性偏慢）。
                # clamp 只约束风格系数本身；调用方的 speed 原样保留，
                # 保证用户始终能手动加速/减速（原先 clamp 会一起压住 speed）。
                # cross_lingual 引擎的风格改用 _STYLE_CL_SPEED
                #（instruct2 的补偿系数是在带指令的条件下标定的，不通用）。
                _engine = self._engine_of(style_resolved)
                if _engine == "cross_lingual" and style_resolved in _STYLE_CL_SPEED:
                    style_factor = _STYLE_CL_SPEED[style_resolved]
                else:
                    style_factor = _STYLE_SPEED.get(style_resolved, 1.0)
                style_factor = max(_SPEED_MIN, min(_SPEED_MAX, style_factor))
                speed_eff = speed * style_factor
                # 参考按风格检索（确定性轮换），并由 seed 决定候选内位置，
                # 既避免随机漂移，也保留同风格下的多样性。
                ref_audio, ref_text = self.pick_ref(lang, style_resolved, pick=seed)
                diag.write(f"[synth] style_resolved={style_resolved} "
                           f"engine={_engine} "
                           f"speed_in={speed} speed_eff={speed_eff:.3f} "
                           f"ref={os.path.basename(ref_audio)} ref_text={ref_text[:40]!r}\n")
                all_chunks = []
                segs = [s for s in _split_text(text, lang) if s.strip()]
                pause = _STYLE_PAUSE.get(style_resolved, 0.22)
                # 超长文本超出 LLM 单次生成能力（提前终止 → 近静音），按句切段逐段合成。
                # 仅日语路径：中文（inference_zero_shot）已随 zh 支持一并移除。
                sys.path.insert(0, WORKDIR)
                from ja_katakana import ja_to_katakana  # noqa: E402
                for si, seg in enumerate(segs):
                    katakana = ja_to_katakana(seg)
                    diag.write(f"[synth] seg_len={len(seg)} katakana_len={len(katakana)} "
                               f"seg_head={seg[:60]!r}\n")
                    seg_chunks = self._synth_seg_guarded(
                        katakana,
                        style_resolved, ref_audio, speed_eff,
                        base_seed=seed + si * 100, n_chars=len(katakana), diag=diag)
                    all_chunks.extend(seg_chunks)
                    # 段间停顿按风格变化（呼吸感）：原先所有风格一律 0.22s，
                    # 会让所有情绪的节奏完全一样，ASMR 场景尤其失真。
                    if si < len(segs) - 1 and seg_chunks:
                        all_chunks.append(torch.zeros(
                            1, int(pause * self.cosyvoice.sample_rate),
                            dtype=seg_chunks[0].dtype))
                chunks = all_chunks
                diag.write(f"[synth] pause={pause} chunks={len(chunks)} "
                           f"chunk_frames={[c.shape[1] for c in chunks]}\n")
                audio = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0)
                diag.write(f"[synth] audio_frames={audio.shape[1]}\n")
                # 兜底：全部块为空（极端 0 帧场景）时产出极短静音，避免空 wav。
                if audio.shape[1] == 0:
                    audio = torch.zeros(1, 100, dtype=audio.dtype)
                    diag.write("[synth] audio empty -> fallback silence\n")
                # 耳语化后处理（可选）：切分在此处，因为要在 torch 域外加噪声/随机相位。
                # ⚠ 实测失败（详见 _WHISPER_ENABLED 处说明），默认关闭。
                if _WHISPER_ENABLED and whisper_mix and whisper_mix > 0:
                    try:
                        audio = self._apply_whisperize(
                            audio, whisper_mix, whisper_lifter, diag, seed)
                    except Exception as e:
                        diag.write(f"[synth] whisperize 失败（跳过）：{e}\n")
                        print(f"[tts-server] whisperize 失败（跳过）：{e}", flush=True)
            finally:
                diag.close()
                self.last_use = time.time()

        fd, wavpath = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            torchaudio.save(wavpath, audio.cpu(), self.cosyvoice.sample_rate)
            with open(wavpath, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(wavpath)
            except OSError:
                pass


def to_mp3(wav: bytes) -> bytes:
    fdin, wavin = tempfile.mkstemp(suffix=".wav")
    os.close(fdin)
    with open(wavin, "wb") as f:
        f.write(wav)
    wavout = wavin + ".mp3"
    try:
        _kw = {}
        if _IS_WIN:   # CREATE_NO_WINDOW 仅 Windows 存在
            _kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run([FFMPEG, "-y", "-i", wavin, "-acodec", "libmp3lame", "-ab", "192k", wavout],
                       check=True, capture_output=True, timeout=120, **_kw)
        with open(wavout, "rb") as f:
            return f.read()
    finally:
        for p in (wavin, wavout):
            try:
                os.remove(p)
            except OSError:
                pass


class Handler(BaseHTTPRequestHandler):
    server_version = "CompanionTTS/0.1"

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _audio(self, data, mime):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/healthz"):
            self._json(200, {"ok": True, "device": "cuda" if TTS.on_gpu else "cpu",
                             "current_lang": TTS.current_lang,
                             "styles": sorted(_STYLE_INSTRUCT_JP.keys()),
                             "ref_durations": float(_STYLE_PAUSE.get("neutral", 0.22))})
        elif self.path.rstrip("/") == "/styles":
            self._json(200, {
                "styles": {
                    k: {"instruct": v,
                        "ref_pref": _STYLE_REF_PREF.get(k, []),
                        "pause": _STYLE_PAUSE.get(k, 0.22)}
                    for k, v in _STYLE_INSTRUCT_JP.items()
                },
                "pool": {s: len(v) for s, v in TTS._vocal_by_style.items()},
                "whisperize": {
                    "param": "whisper_mix",
                    "status": "DISABLED",
                    "enabled": _WHISPER_ENABLED,
                    "reason": "实测失败：v1(随机相位)/v2(噪声声码器)/加法式气声全部有严重杂音，"
                              "严重度随注入噪声量上升。诊断确认非削波/直流/采样率问题，"
                              "而是方法概念错误——真耳语是用噪声激励替换声带振动，"
                              "在正常语音上叠加噪声只会得到嘶嘶声。",
                    "alternative": "需要耳语请用 POST /vc",
                },
                "vc": {
                    "endpoint": "POST /vc",
                    "body": {"source": "源音频路径（如别人的耳语素材；可长于 30s）",
                             "prompt": "音色参考音频路径（可选，默认按 style 取池内参考）",
                             "style": "取参考时用的风格（prompt 省略时生效）",
                             "speed": 1.0,
                             "auto_split": "默认 true：长音频按静音边界自动切分再拼接",
                             "deess": "默认 true：输出做去齿音/高频控制",
                             "response_format": "wav|mp3"},
                    "why": "不注入噪声：从 source 提取 speech token（保留内容/韵律/气声结构），"
                           "用 prompt 提供音色，直接交 flow 重建。实测 HNR 保持 -2.5dB"
                           "（source -3.3 / prompt +6.2），气声得以保留。",
                    "deess_detail": {
                        "params": _DEESS_DEFAULT,
                        "note": "只改幅度谱、保留相位，故不会产生 whisperize 那类机械感。"
                                "实测 >8k 由 1.29% 降至 0.30%（真实耳语素材为 0.25%），"
                                "HNR 与 RMS 几乎不变、包络相关 0.997。",
                    },
                },
                "instructions_style_map": {
                    "note": "OpenAI 兼容接口的 instructions 会归类到下列 style；"
                            "显式传 style 时以 style 为准。",
                    "rules": {k: [kw for kw, _ in v]
                              for k, v in _INSTR_STYLE_KEYWORDS.items()},
                },
                "note": "instruct2 只改语速与停顿节奏；RVC 会把气声治回谐波"
                        "（其权重均为 f0=1 音高引导）。真耳语请用 /vc。",
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self._body() or b"{}")
        except Exception:
            return self._json(400, {"error": "invalid json"})

        if self.path.rstrip("/") == "/tts":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json(400, {"error": "text required"})
            lang = body.get("lang") or "auto"
            speed = float(body.get("speed") or 1.0)
            style = body.get("style") or "auto"
            seed = body.get("seed")
            seed = int(seed) if seed not in (None, "") else None
            # whisper_mix: 0=不耳语化；0.85 左右为明显气声；1.0 为最强
            whisper_mix = float(body.get("whisper_mix") or 0.0)
            try:
                wav = TTS.synth(text, lang, speed, style, seed, whisper_mix)
            except UnsupportedLanguage as e:
                # 语言不支持属于客户端错误，返回 400 而非 500
                return self._json(400, {"error": str(e),
                                        "supported_lang": list(LORA.keys())})
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {"error": f"synth failed: {e}"})
            self._audio(wav, "audio/wav")

        elif self.path.rstrip("/") == "/v1/audio/speech":
            text = (body.get("input") or "").strip()
            if not text:
                return self._json(400, {"error": "input required"})
            # 只支持日语：voice/lang 字段保留以兼容 OpenAI 客户端，但不影响语言
            lang = "jp"
            speed = float(body.get("speed") or 1.0)
            fmt = (body.get("response_format") or "mp3").lower()
            # 情绪：显式 style 优先；否则从 OpenAI 风格的 instructions 推断。
            # Hermes 的 text_to_speech 会带 instructions（实测 100% 携带），
            # 这里把它归类到已标定的 style，使 agent 的情绪判断真正生效。
            style = (body.get("style") or "").strip()
            instr = body.get("instructions")
            style_src = "explicit"
            if not style:
                style, score = instructions_to_style(instr)
                style_src = f"instructions(score={score})" if score else "default"
            seed = body.get("seed")
            seed = int(seed) if seed not in (None, "") else None
            whisper_mix = float(body.get("whisper_mix") or 0.0)
            # 记录 instructions 的推断结果，便于事后核对 Hermes 触发了哪种情绪
            try:
                with open(os.path.join(WORKDIR, "tts_synth_diag.log"),
                          "a", encoding="utf-8") as _d:
                    _d.write(f"[openai-tts] style={style} src={style_src} "
                             f"instructions={str(instr)[:120]!r}\n")
            except Exception:
                pass
            try:
                wav = TTS.synth(text, lang, speed, style, seed, whisper_mix)
            except UnsupportedLanguage as e:
                return self._json(400, {"error": str(e),
                                        "supported_lang": list(LORA.keys())})
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {"error": f"synth failed: {e}"})
            if fmt in ("mp3", "mp3_flac"):
                try:
                    self._audio(to_mp3(wav), "audio/mpeg")
                except Exception:
                    self._audio(wav, "audio/wav")
            else:
                self._audio(wav, "audio/wav")

        elif self.path.rstrip("/") == "/vc":
            # 语音转换：把 source（如别人的耳语素材）换成昴的音色。
            # 不注入噪声，实测能保留 source 的气声（HNR 贴近 source）。
            src = (body.get("source") or "").strip()
            if not src:
                return self._json(400, {"error": "source required (wav 路径)"})
            prompt = (body.get("prompt") or "").strip() or None
            speed = float(body.get("speed") or 1.0)
            style = body.get("style") or "neutral"
            auto_split = body.get("auto_split")
            auto_split = True if auto_split is None else bool(auto_split)
            deess = body.get("deess")
            deess = _DEESS_ENABLED if deess is None else bool(deess)
            fmt = (body.get("response_format") or "wav").lower()
            try:
                wav = TTS.vc(src, prompt, speed, style, auto_split, deess=deess)
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {"error": f"vc failed: {e}"})
            if fmt in ("mp3", "mp3_flac"):
                try:
                    self._audio(to_mp3(wav), "audio/mpeg")
                except Exception:
                    self._audio(wav, "audio/wav")
            else:
                self._audio(wav, "audio/wav")

        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        print("[tts-server]", fmt % args, flush=True)


TTS = None  # 在 main() 里按 --device 创建


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0", help="监听地址；0.0.0.0 = 局域网可访问（工位/笔记本共用）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto=CPU常驻/用时搬GPU；cpu=纯CPU（不占显卡）；cuda=一直留GPU")
    ap.add_argument("--gpu-idle-seconds", type=int, default=60,
                    help="空闲多少秒后把模型退回 CPU；0 = 一直留在 GPU")
    args = ap.parse_args()

    global GPU_IDLE_SECONDS, TTS
    GPU_IDLE_SECONDS = args.gpu_idle_seconds

    TTS = TTSService(device=args.device)
    TTS.ensure_loaded("jp")     # 只预热日语（中文支持已移除）
    if args.device == "cuda":
        TTS.to_gpu()

    def watcher():
        while True:
            time.sleep(5)
            TTS.idle_tick()

    threading.Thread(target=watcher, daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[tts-server] 监听 http://{args.host}:{args.port}  (gpu_idle_seconds={GPU_IDLE_SECONDS})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
