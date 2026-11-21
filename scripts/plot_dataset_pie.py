#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

from filterguard.data import load_rows


def _count_types(rows: list[dict[str, Any]]) -> dict[str, int]:
    normal = polluted = poisoned = both = 0
    for ex in rows:
        is_polluted = bool(ex.get("is_polluted"))
        is_poisoned = bool(ex.get("is_poisoned"))
        if is_polluted and is_poisoned:
            both += 1
        if is_poisoned:
            poisoned += 1
        elif is_polluted:
            polluted += 1
        else:
            normal += 1
    return {
        "normal": normal,
        "polluted": polluted,
        "poisoned": poisoned,
        "both": both,
        "total": normal + polluted + poisoned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pie chart for dataset distribution")
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument(
        "--output",
        default="docs/presentation_ver/dataset_v0_1_1_pie.png",
        help="Output path (.png)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Missing matplotlib. Install: pip install matplotlib") from e

    rows = load_rows(str(args.dataset_path), limit=0)
    counts = _count_types(rows)
    total = counts["total"]

    if total <= 0:
        raise ValueError("empty dataset: total=0")

    labels = ["正常", "污染", "投毒"]
    values = [counts["normal"], counts["polluted"], counts["poisoned"]]
    colors = ["#4CAF50", "#FFC107", "#F44336"]

    # 过滤掉数量为0的部分
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for l, v, c in zip(labels, values, colors):
        if v > 0:
            filtered_labels.append(f"{l}\n(n={v})")
            filtered_values.append(v)
            filtered_colors.append(c)

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(5, 5))

    def autopct(pct: float) -> str:
        return f"{pct:.1f}%"

    wedges, texts, autotexts = ax.pie(
        filtered_values,
        labels=filtered_labels,
        colors=filtered_colors,
        startangle=90,
        autopct=autopct,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 16, "fontweight": "bold"},
    )

    # 设置百分比文字颜色为白色，增加对比度
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(16)

    ax.axis("equal")
    fig.tight_layout()

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
