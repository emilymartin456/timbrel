"""韵律迁移示例：用 A 的音色 + B 的韵律合成同一段文本。

同样使用未训练的小模型演示接口。

运行：
    python examples/prosody_transfer.py
"""

import torch

from timbrel import ModelConfig, build_model
from timbrel.frontend import PhonemeVocab
from timbrel.frontend.bilingual import BilingualFrontend
from timbrel.prosody import transfer_prosody


def main() -> None:
    torch.manual_seed(0)
    vocab = PhonemeVocab()
    frontend = BilingualFrontend(vocab)
    config = ModelConfig(vocab_size=len(vocab), hidden=64, n_enc_layers=2, n_dec_layers=2)
    model = build_model(config)
    model.eval()

    ids = frontend.encode("This is a prosody transfer demo. 这是韵律迁移演示。")
    phonemes = torch.tensor([ids])
    src_lengths = torch.tensor([len(ids)])

    timbre_ref = torch.randn(1, 70, config.n_mels)  # 说话人 A：提供音色
    prosody_ref = torch.randn(1, 90, config.n_mels)  # 说话人 B：提供韵律
    mel = transfer_prosody(model, phonemes, src_lengths, timbre_ref, prosody_ref)
    print("迁移后 mel 形状:", tuple(mel.shape))


if __name__ == "__main__":
    main()
