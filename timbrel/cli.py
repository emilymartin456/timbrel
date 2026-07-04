"""Command line interface: ``timbrel <command>``.

Deliberately imports only the (torch-free) front-end so ``timbrel phonemize``
and ``timbrel symbols`` start instantly without loading the model stack.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from timbrel.config import Config
from timbrel.frontend.bilingual import BilingualFrontend
from timbrel.frontend.phonemes import PhonemeVocab
from timbrel.version import __version__


def _cmd_info(args: argparse.Namespace) -> int:
    cfg = Config()
    print(f"timbrel {__version__}")
    print(f"sample_rate={cfg.audio.sample_rate} n_mels={cfg.audio.n_mels}")
    print(f"vocab_size={len(PhonemeVocab())}")
    return 0


def _cmd_symbols(args: argparse.Namespace) -> int:
    print(" ".join(PhonemeVocab().symbols))
    return 0


def _cmd_phonemize(args: argparse.Namespace) -> int:
    frontend = BilingualFrontend()
    print(" ".join(frontend.to_phonemes(args.text, language=args.language)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="timbrel", description="双语多说话人声音克隆声学模型")
    parser.add_argument("--version", action="version", version=f"timbrel {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_info = sub.add_parser("info", help="打印版本与默认配置")
    p_info.set_defaults(func=_cmd_info)

    p_symbols = sub.add_parser("symbols", help="打印音素表")
    p_symbols.set_defaults(func=_cmd_symbols)

    p_phon = sub.add_parser("phonemize", help="将文本转换为音素序列")
    p_phon.add_argument("text")
    p_phon.add_argument("--language", "-l", default="auto", choices=["auto", "zh", "en"])
    p_phon.set_defaults(func=_cmd_phonemize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
