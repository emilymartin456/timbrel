# 架构总览

Timbrel 是一个非自回归（FastSpeech2 风格）的多说话人声学模型。核心思路是把一段文本的**内容**、目标说话人的**音色**、参考语音的**韵律**三者分离，再在解码阶段重新组合，从而支持少样本克隆与韵律迁移。

## 数据流

训练阶段（参考音频通常就是目标句本身）：

```
phonemes ──► TextEncoder ─────────────► content（说话人无关）
                                             │
ref_mel  ──► SpeakerEncoder ─► speaker（音色 d-vector）
ref_mel  ──► ProsodyEncoder ─► prosody + FiLM(gamma, beta)
                                             │
              content ──FiLM(prosody)──► VarianceAdaptor ──► LengthRegulator ──► MelDecoder(speaker) ──► mel
```

解耦相关的头（对抗说话人分类器 + CLUB 互信息）作用在池化后的 content 与 prosody 上。

## 模块

| 模块 | 文件 | 职责 |
|---|---|---|
| 双语前端 | `frontend/` | 文本 → 音素 id（中文 pypinyin + 声调，英文词典/规则，中英混排切分）|
| 文本编码器 | `models/text_encoder.py` | 音素嵌入 + FFT block 堆叠；末端实例归一化形成内容瓶颈 |
| 说话人编码器 | `models/speaker_encoder.py` | 参考 mel → L2 归一化的 d-vector（音色）|
| 韵律编码器 | `models/prosody_encoder.py` | 参考 mel → 韵律向量 + FiLM 调制参数 |
| 变长适配器 | `models/variance.py` | 时长/基频/能量预测（音素级）+ 长度规整 |
| 解码器 | `models/decoder.py` | 条件 LayerNorm 注入音色，输出 mel + PostNet 精修 |
| 解耦 | `models/disentangle.py` | 梯度反转、对抗说话人分类器、CLUB 互信息上界 |

## 张量约定

- 音素序列：`(B, T_text)`，padding mask `True` 表示补位。
- mel 谱：模型内部统一为 `(B, T, n_mels)`；`audio.MelSpectrogram` 抽取结果为 `(B, n_mels, T)`，进模型前转置。
- 说话人嵌入：`(B, speaker_dim)`；韵律嵌入：`(B, prosody_dim)`。
- 时长：`(B, T_text)`，长度规整后帧数 `sum(durations)`。

## 损失

`losses.TimbrelLoss` 汇总：mel L1（含 PostNet）、时长/基频/能量 MSE、对抗说话人交叉熵、CLUB 互信息、FiLM 正则，按 `TrainConfig` 中的权重加权。CLUB 变分网络另有自身的似然目标，由 `Trainer` 单独优化。
