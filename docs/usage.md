# 使用指南

## 安装

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## 文本前端

```python
from timbrel import BilingualFrontend

fe = BilingualFrontend()
fe.to_phonemes("你好，世界。")          # 音素符号列表
fe.encode("你好 world")                # 音素 id 列表
fe.to_phonemes("绿色", language="zh")   # 指定语言，跳过自动切分
```

- 中文：`pypinyin` 转 tone3 拼音 → 声母 + 带调韵母，并做三声变调近似。
- 英文：内置词典优先，未登录词走字母到 ARPAbet 的规则回退。
- 自动模式按 Unicode 区间切分中英，标点转换为停顿标记（`sp` / `sil`）。

## 构建模型

```python
from timbrel import ModelConfig, build_model, load_config

config = load_config("configs/base.yaml")        # 或直接 ModelConfig(...)
model = build_model(ModelConfig(vocab_size=350))
```

## 训练

```python
from torch.utils.data import DataLoader
from timbrel import Config
from timbrel.data import SpeechDataset, collate_fn
from timbrel.trainer import Trainer

dataset = SpeechDataset(samples)   # 每个 sample: phonemes/mel/durations/pitch/energy/speaker_id
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

trainer = Trainer(model, Config())
trainer.fit(loader, max_steps=1000)
```

特征抽取（mel / f0 / 能量）见 `timbrel.audio`，建议离线预处理后存盘。

## 推理与克隆

```python
from timbrel.inference import Synthesizer

synth = Synthesizer(model)                  # 或 Synthesizer.from_checkpoint("ckpt.pt")
mel = synth.clone("很高兴认识你。", ref_mel)  # ref_mel: (T, n_mels)
```

## 少样本自适应

```python
from timbrel.adaptation import SpeakerEnroller, adapt, freeze_for_adaptation

# 方式一：直接对参考 d-vector 取平均（零训练）
speaker = SpeakerEnroller(model).enroll([ref1, ref2, ref3])

# 方式二：只微调条件 LayerNorm 参数
freeze_for_adaptation(model)
adapt(model, batches, steps=100, lr=1e-4)
```

## 韵律迁移

```python
from timbrel.prosody import transfer_prosody

mel = transfer_prosody(model, phonemes, lengths, timbre_ref, prosody_ref)
```

## 命令行

```bash
timbrel info                          # 版本与配置
timbrel symbols                       # 打印音素表
timbrel phonemize "你好 world" -l auto
```
