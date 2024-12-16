"""把中英混合文本转换为音素序列与整数编号。

运行：
    python examples/phonemize_bilingual.py
"""

from timbrel.frontend.bilingual import BilingualFrontend

SENTENCES = [
    "你好，世界。",
    "我在 Wuhan 大学做 voice cloning 研究。",
    "Timbrel 支持中英双语，例如 3 个说话人。",
]


def main() -> None:
    frontend = BilingualFrontend()
    for text in SENTENCES:
        phones = frontend.to_phonemes(text)
        ids = frontend.encode(text)
        print(f"文本 : {text}")
        print(f"音素 : {' '.join(phones)}")
        print(f"编号 : {ids}")
        print("-" * 48)
    print(f"音素表大小: {len(frontend.vocab)}")


if __name__ == "__main__":
    main()
