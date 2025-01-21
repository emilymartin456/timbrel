# API 参考

只列出稳定性相对较高的公开接口；内部模块（`models/*` 的具体层）以源码为准。

## 顶层导出（`timbrel`）

| 名称 | 说明 |
|---|---|
| `__version__` | 版本号字符串 |
| `Config` / `ModelConfig` | 配置数据类 |
| `load_config(path)` / `save_config(cfg, path)` | YAML 读写 |
| `BilingualFrontend` | 中英双语文本前端 |
| `PhonemeVocab` | 音素 ↔ id 词表 |
| `AcousticModel` / `build_model(config)` | 声学模型与工厂 |

## 前端（`timbrel.frontend`）

- `BilingualFrontend(vocab=None, config=None)`
  - `to_phonemes(text, language=None) -> list[str]`
  - `encode(text, language=None, add_bos_eos=None) -> list[int]`
- `PhonemeVocab(symbols=None)`：`encode` / `decode` / `pad_id` / `unk_id` 等。
- `ChineseG2P` / `EnglishG2P`：`to_phonemes(text)`。

## 音频（`timbrel.audio`）

- `MelSpectrogram(config=None)(wav) -> (B, n_mels, T)`
- `mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)`
- `extract_f0(wav, sample_rate, hop_length, ...) -> np.ndarray`
- `frame_energy(log_mel) -> np.ndarray`
- `average_by_duration(values, durations, voiced_only=False)`

## 模型（`timbrel.models`）

- `AcousticModel(config)`
  - `forward(phonemes, src_lengths, ref_mel, ref_lengths=None, durations=None, pitch=None, energy=None, max_mel_len=None) -> dict`
  - `infer(phonemes, src_lengths, speaker_ref_mel, prosody_ref_mel=None, ...) -> (mel, stats)`

## 自适应与韵律

- `timbrel.adaptation.SpeakerEnroller(model).enroll(ref_mels) -> (1, speaker_dim)`
- `timbrel.adaptation.freeze_for_adaptation(model) -> list[Parameter]`
- `timbrel.adaptation.adapt(model, batches, criterion=None, steps=50, lr=1e-4) -> list[float]`
- `timbrel.prosody.transfer_prosody(model, phonemes, src_lengths, timbre_ref_mel, prosody_ref_mel, ...) -> mel`

## 训练与推理

- `timbrel.losses.TimbrelLoss(config=None)(outputs, targets) -> (total, breakdown)`
- `timbrel.trainer.Trainer(model, config, device="cpu")`：`train_step(batch)` / `fit(loader, max_steps=None)`
- `timbrel.inference.Synthesizer(model, frontend=None, config=None)`：`clone(text, ref_mel, ...)` / `from_checkpoint(path)`
- `timbrel.data.SpeechDataset(samples)` / `collate_fn(batch)` / `parse_filelist(path)`
