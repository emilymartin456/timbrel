"""少样本声音克隆的最小示例。

用随机初始化的小模型 + 随机 mel 演示「注册 → 克隆」接口，未训练，输出不可听。
真实使用时请加载训练好的权重，并用真实音频提取的 mel 作为参考。

运行：
    python examples/clone_fewshot.py
"""

import torch

from timbrel import ModelConfig, build_model
from timbrel.adaptation import SpeakerEnroller
from timbrel.frontend import PhonemeVocab
from timbrel.frontend.bilingual import BilingualFrontend
from timbrel.inference import Synthesizer


def main() -> None:
    torch.manual_seed(0)
    vocab = PhonemeVocab()
    config = ModelConfig(vocab_size=len(vocab), hidden=64, n_enc_layers=2, n_dec_layers=2)
    model = build_model(config)
    model.eval()

    # 目标说话人的 3 段参考 mel，形状 (T, n_mels)
    references = [torch.randn(70, config.n_mels) for _ in range(3)]
    speaker_embedding = SpeakerEnroller(model).enroll(references)
    print("说话人 d-vector 形状:", tuple(speaker_embedding.shape))

    synth = Synthesizer(model, BilingualFrontend(vocab))
    mel = synth.clone("你好，很高兴认识你。", references[0])
    print("合成 mel 形状:", tuple(mel.shape))


if __name__ == "__main__":
    main()
