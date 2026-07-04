# Timbrel

[![CI](https://github.com/emilymartin456/timbrel/actions/workflows/ci.yml/badge.svg)](https://github.com/emilymartin456/timbrel/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20--%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Timbrel** 是一个面向研究的**多说话人声音克隆声学模型**：在非自回归（FastSpeech2 风格）骨干之上，围绕「**说话人自适应 + 音色解耦 + 少样本克隆 + 韵律迁移**」几个方向做基础实现，并原生支持**中英双语**前端。

> 🚧 研究早期项目，接口尚未稳定。目前聚焦于「文本 + 参考音频 → 梅尔谱」这一声学建模阶段，声码器（mel → 波形）不在当前范围内。

## 特性

- **音色解耦**：内容编码器用实例归一化去除句级统计，并配合梯度反转的对抗说话人分类器与 CLUB 互信息上界，抑制内容表征中的说话人信息。
- **说话人自适应**：解码器采用**条件 LayerNorm**（AdaSpeech 风格）注入音色；自适应时只微调这部分参数，天然适合少样本。
- **少样本克隆**：对几段参考语音的 d-vector 取平均即可完成说话人注册（enrollment）。
- **韵律迁移**：独立的韵律编码器输出 FiLM 调制参数，且经对抗训练与说话人解耦，可将 A 的音色与 B 的韵律组合。
- **中英双语前端**：中文经 `pypinyin` 转声母/韵母 + 声调（含三声变调），英文走内置词典 + 规则回退，统一到同一音素表；支持中英混排（code-switch）。
- **依赖精简**：仅 `torch` / `numpy` / `pypinyin` / `pyyaml`，不依赖 librosa —— 梅尔谱与 f0 自带实现。

## 安装

```bash
# CPU 版 PyTorch（如需 GPU 请替换为对应 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## 快速开始

### 文本转音素（中英双语）

```python
from timbrel import BilingualFrontend

fe = BilingualFrontend()
fe.to_phonemes("你好，世界。")            # ['n', 'i2', 'h', 'ao3', ...]
fe.to_phonemes("我在做 voice cloning。")  # 中英混排自动切分
```

命令行同样可用：

```bash
timbrel phonemize "你好 world" --language auto
```

### 少样本克隆

```python
import torch
from timbrel import ModelConfig, build_model
from timbrel.adaptation import SpeakerEnroller
from timbrel.inference import Synthesizer

model = build_model(ModelConfig(vocab_size=350))
refs = [torch.randn(70, 80) for _ in range(3)]     # 目标说话人的参考 mel (T, n_mels)
speaker = SpeakerEnroller(model).enroll(refs)       # 少样本注册

synth = Synthesizer(model)
mel = synth.clone("很高兴认识你。", refs[0])          # (T, n_mels)
```

### 韵律迁移

```python
from timbrel.prosody import transfer_prosody

mel = transfer_prosody(model, phonemes, lengths, timbre_ref, prosody_ref)
# timbre_ref 提供音色，prosody_ref 提供韵律
```

更多示例见 [`examples/`](examples/)。

## 文档

- [架构总览](docs/architecture.md)
- [使用指南](docs/usage.md)
- [设计笔记（解耦思路）](docs/design-notes.md)
- [API 参考](docs/api-reference.md)

## 许可证

[MIT](LICENSE) © 2026 Jiang Que
