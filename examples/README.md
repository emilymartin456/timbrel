# 示例

每个脚本都可独立运行，用极小的随机模型演示接口（真实使用时请替换为训练好的权重与真实音频提取的 mel）。

```bash
python examples/phonemize_bilingual.py   # 中英双语文本 → 音素
python examples/clone_fewshot.py         # 少样本说话人注册 + 克隆
python examples/prosody_transfer.py      # 韵律迁移：A 的音色 + B 的韵律
```
