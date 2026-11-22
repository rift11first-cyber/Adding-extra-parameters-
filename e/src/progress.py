from __future__ import annotations

import sys
from tqdm import tqdm

BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def bar(total: int | None, desc: str, unit: str = "it", leave: bool = True) -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        bar_format=BAR_FORMAT,
    )
