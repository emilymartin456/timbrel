"""Parsing of pipe-delimited metadata filelists.

Each non-empty, non-comment line is ``utt_id|text|audio_path|speaker``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


@dataclass
class MetaItem:
    utt_id: str
    text: str
    audio_path: str
    speaker: str


def parse_filelist(path: PathLike) -> list[MetaItem]:
    items: list[MetaItem] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            raise ValueError(f"expected 4 fields (utt|text|audio|speaker), got {line!r}")
        items.append(MetaItem(parts[0], parts[1], parts[2], parts[3]))
    return items
