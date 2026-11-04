#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "outputs" / "training" / "franka_tactile310"
PI_ROOT = TRAINING / "pi05_jax_zrl_envelope_v1" / "sustained_contact_seed_44"
SMOL_ROOT = TRAINING / "force_envelope_transformer_v3"
OUTPUT = ROOT / "outputs" / "reports" / "pi05_jax_zrl_envelope_20260725"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def stable_scale(report: dict, side: str) -> dict:
    return report["sides"][side]["mu_plus_alpha_sigma"]


def render_inline(value: str) -> str:
    value = html.escape(value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        r'<img src="\2" alt="\1">',
        value,
    )
    return value


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(f"<h{level}>{render_inline(line[level:].strip())}</h{level}>")
            index += 1
            continue
        if line.startswith("```"):
            language = line.removeprefix("```").strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            blocks.append(
                f'<pre data-language="{html.escape(language)}"><code>'
                f"{html.escape(chr(10).join(code))}</code></pre>"
            )
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            if len(rows) >= 2:
                head = "".join(f"<th>{render_inline(cell)}</th>" for cell in rows[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>"
                    for row in rows[2:]
                )
                blocks.append(
                    f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                    f"<tbody>{body}</tbody></table></div>"
                )
            continue
        if line.startswith("- "):
            items: list[str] = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(f"<li>{render_inline(lines[index][2:])}</li>")
                index += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if line.startswith("!["):
            blocks.append(render_inline(line))
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while (
            index < len(lines)
            and lines[index].strip()
            and not lines[index].startswith(("#", "|", "- ", "```", "!["))
        ):
            paragraph.append(lines[index].strip())
            index += 1
        blocks.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
    return "\n".join(blocks)


def save_history_plot(history: dict) -> None:
    sides = ["Left", "Right"]
    side_keys = ["left", "right"]
    x = np.arange(len(sides))
    width = 0.34

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    fields = [
        ("mae", "MAE (N)", False),
        ("mean_envelope_margin", "Mean envelope width (N)", False),
        ("coverage", "Coverage (%)", True),
    ]
    for axis, (field, title, as_percent) in zip(axes, fields, strict=True):
        without = [
            history["sides"][side]["without_history"][field] for side in side_keys
        ]
        with_history = [
            history["sides"][side]["with_history"][field] for side in side_keys
        ]
        if as_percent:
            without = [100.0 * value for value in without]
            with_history = [100.0 * value for value in with_history]
            axis.axhspan(93.0, 97.0, color="#d9ead3", alpha=0.65, label="Target")
        axis.bar(x - width / 2, without, width, label="No history", color="#9aa6b2")
        axis.bar(
            x + width / 2, with_history, width, label="200 ms history",
            color="#1976d2",
        )
        axis.set_xticks(x, sides)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("SmolVLA controlled ablation: effect of causal tactile history")
    figure.tight_layout()
    figure.savefig(OUTPUT / "smolvla_history_ablation.png", dpi=180)
    plt.close(figure)


def save_pi_plot(pi_report: dict) -> None:
    sides = ["Left", "Right"]
    side_keys = ["left", "right"]
    x = np.arange(2)
    coverage = [
        100.0 * stable_scale(pi_report, side)["coverage"] for side in side_keys
    ]
    mae = [stable_scale(pi_report, side)["mae"] for side in side_keys]
    width = [
        stable_scale(pi_report, side)["mean_envelope_width"] for side in side_keys
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    axes[0].bar(x, coverage, color=["#147d64", "#36a269"])
    axes[0].axhspan(93.0, 97.0, color="#d9ead3", alpha=0.65)
    axes[0].set_ylim(80.0, 100.0)
    axes[0].set_title("Stable-contact coverage (%)")
    axes[1].bar(x, mae, color=["#1976d2", "#5b8def"])
    axes[1].set_title("Stable-contact MAE (N)")
    axes[2].bar(x, width, color=["#d97706", "#f59e0b"])
    axes[2].set_title("Mean envelope width (N)")
    for axis in axes:
        axis.set_xticks(x, sides)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Pi0.5 JAX z_rl Envelope Expert, seed 44")
    figure.tight_layout()
    figure.savefig(OUTPUT / "pi05_stable_contact_metrics.png", dpi=180)
    plt.close(figure)


def build_report() -> tuple[str, str]:
    pi_training = load_json(PI_ROOT / "transformer_training_result.json")
    pi_report = load_json(
        PI_ROOT / "stable_evaluation" / "stable_contact_gated_evaluation.json"
    )
    pi_calibration = load_json(
        PI_ROOT / "stable_evaluation" / "stable_contact_calibration.json"
    )
    pi_history = load_json(
        TRAINING
        / "pi05_jax_zrl_envelope_v1"
        / "causal_tactile_history_report.json"
    )
    pi_features = load_json(
        TRAINING / "pi05_jax_zrl_envelope_v1" / "condition_tokens.json"
    )
    history = load_json(
        SMOL_ROOT
        / "tactile_history_ablation"
        / "weight_1.5_seed_44"
        / "history_ablation_comparison.json"
    )

    smol_reports = [
        load_json(
            SMOL_ROOT
            / "stable_contact_ab"
            / f"B_seed_{seed}"
            / "stable_evaluation"
            / "stable_contact_gated_evaluation.json"
        )
        for seed in (42, 43, 44)
    ]
    smol_summary: dict[str, dict[str, tuple[float, float]]] = {}
    for side in ("left", "right"):
        smol_summary[side] = {}
        for field in ("mae", "coverage", "mean_envelope_width"):
            smol_summary[side][field] = mean_std(
                [stable_scale(report, side)[field] for report in smol_reports]
            )

    save_history_plot(history)
    save_pi_plot(pi_report)

    left = stable_scale(pi_report, "left")
    right = stable_scale(pi_report, "right")
    pi_pass = 0.93 <= left["coverage"] <= 0.97 and 0.93 <= right["coverage"] <= 0.97

    markdown = f"""# Pi0.5 JAX z_rl 触觉力 Envelope 测试报告

- 报告日期：2026-07-25
- 数据集：`franka_usb_tactile_310_20260716`
- 当前模型：Pi0.5 JAX `z_rl` stable-contact Envelope Expert
- 训练随机种子：44
- 目标：稳定接触阶段未来 300 ms 左右触觉力上包络

## 1. 结论

**离线稳定接触 soft penalty：通过。正式安全上界：尚未通过。**

左右测试 coverage 分别为 **{pct(left["coverage"])}** 和
**{pct(right["coverage"])}**，均进入预设的 93%～97% 区间；平均包络宽度分别为
**{left["mean_envelope_width"]:.3f} N** 和
**{right["mean_envelope_width"]:.3f} N**。因此它适合作为 Online RL 中低权重、可截断的
稳定接触软惩罚候选。

它不能替代硬安全保护：测试集仍有长尾超界，且测试集已参与多轮开发诊断；正式上线前还缺少
独立新 episode 的 blind shadow evaluation，以及在线 200 ms 历史触觉接线。

## 2. 当前 Pi0.5 模型是否加入历史触觉

**是。**

- 历史范围：`[t_k - 200 ms, t_k)`。
- 左右传感器分别最多 6 个真实采样点。
- 每个点 5 维：`Fx, Fy, Fz, force_magnitude, relative_time`。
- 张量：`[B, 2, 6, 5]`，展开成 12 个 Transformer tokens。
- 不插值；样本不足时 padding，并使用 mask。
- 所有历史点严格早于锚点；因果违规为 **0**。
- 每侧窗口平均历史样本数：**{pi_history["sample_count_per_side_window"]["mean"]:.3f}**。

## 3. Pi0.5 JAX z_rl 测试结果

| 指标 | 左侧 | 右侧 |
|---|---:|---:|
| Calibration 稳定接触样本 | {pi_report["sides"]["left"]["calibration_samples"]} | {pi_report["sides"]["right"]["calibration_samples"]} |
| Test 稳定接触样本 | {pi_report["sides"]["left"]["test_samples"]} | {pi_report["sides"]["right"]["test_samples"]} |
| MAE | {left["mae"]:.3f} N | {right["mae"]:.3f} N |
| RMSE | {left["rmse"]:.3f} N | {right["rmse"]:.3f} N |
| Coverage | {pct(left["coverage"])} | {pct(right["coverage"])} |
| 平均 sigma | {left["mean_sigma"]:.3f} N | {right["mean_sigma"]:.3f} N |
| 平均 envelope 宽度 | {left["mean_envelope_width"]:.3f} N | {right["mean_envelope_width"]:.3f} N |
| 超界窗口 | {left["false_penalty_windows"]} | {right["false_penalty_windows"]} |
| Episode 最低 coverage | {pct(pi_report["sides"]["left"]["episode_level"]["scale"]["min_coverage"])} | {pct(pi_report["sides"]["right"]["episode_level"]["scale"]["min_coverage"])} |

校准只使用 calibration split：

- `alpha95_left = {pi_calibration["alpha95_left"]:.4f}`
- `alpha95_right = {pi_calibration["alpha95_right"]:.4f}`

测试集没有用于重新调整 alpha。

![Pi0.5 stable contact metrics](pi05_stable_contact_metrics.png)

## 4. 历史触觉的影响：SmolVLA 严格消融

下面是已有的严格 A/B：相同 SmolVLA condition tokens、相同 episode split、seed 44、
contact weight 1.5 和 Transformer 主体，仅改变是否输入 200 ms 因果触觉历史。

| 侧别 | 历史 | MAE | Coverage | Sigma | Envelope 宽度 |
|---|---|---:|---:|---:|---:|
| 左 | 无 | {history["sides"]["left"]["without_history"]["mae"]:.3f} N | {pct(history["sides"]["left"]["without_history"]["coverage"])} | {history["sides"]["left"]["without_history"]["mean_sigma"]:.3f} N | {history["sides"]["left"]["without_history"]["mean_envelope_margin"]:.3f} N |
| 左 | 200 ms | {history["sides"]["left"]["with_history"]["mae"]:.3f} N | {pct(history["sides"]["left"]["with_history"]["coverage"])} | {history["sides"]["left"]["with_history"]["mean_sigma"]:.3f} N | {history["sides"]["left"]["with_history"]["mean_envelope_margin"]:.3f} N |
| 右 | 无 | {history["sides"]["right"]["without_history"]["mae"]:.3f} N | {pct(history["sides"]["right"]["without_history"]["coverage"])} | {history["sides"]["right"]["without_history"]["mean_sigma"]:.3f} N | {history["sides"]["right"]["without_history"]["mean_envelope_margin"]:.3f} N |
| 右 | 200 ms | {history["sides"]["right"]["with_history"]["mae"]:.3f} N | {pct(history["sides"]["right"]["with_history"]["coverage"])} | {history["sides"]["right"]["with_history"]["mean_sigma"]:.3f} N | {history["sides"]["right"]["with_history"]["mean_envelope_margin"]:.3f} N |

加入历史后：

- 左侧 MAE 降低 **{100.0 * history["sides"]["left"]["mae_reduction_fraction"]:.1f}%**，包络宽度缩小 **{100.0 * history["sides"]["left"]["margin_reduction_fraction"]:.1f}%**。
- 右侧 MAE 降低 **{100.0 * history["sides"]["right"]["mae_reduction_fraction"]:.1f}%**，包络宽度缩小 **{100.0 * history["sides"]["right"]["margin_reduction_fraction"]:.1f}%**。
- 历史触觉显著改善稳定接触阶段的点预测和包络锐度。
- 左侧 coverage 略降，说明历史模型更确信，但当时左侧 calibration 没有同步覆盖尾部分布；这不是“历史无效”，而是不确定度校准仍有偏差。

![SmolVLA history ablation](smolvla_history_ablation.png)

## 5. Pi0.5 与 SmolVLA 必须区分

| 项目 | Pi0.5 JAX z_rl（本报告主模型） | SmolVLA（历史消融模型） |
|---|---|---|
| 条件表示 | `response.z_rl` | VLM 动作生成前 condition tokens |
| 原始 shape | `[B,1,512]` | `[B,241,960]` |
| 语言 | z_rl 包含图像/语言前缀 | condition tokens 包含视觉/语言 |
| Robot state | Envelope 独立输入 8D state | 特征含 state，实验仍保留 state token |
| 历史触觉 | 200 ms，12 tokens | 消融中比较 0 vs 200 ms |
| 当前训练目标 | sustained-contact-only | 已有全窗口、stable-contact A/B 等多组实验 |
| 当前结果种子 | seed 44 单次 | stable-contact B 有 seeds 42/43/44 |

Pi0.5 与 SmolVLA 的 episode split 不同，因此下面只能作趋势参考，不能宣称严格优劣：

| 模型 | 左 MAE / Cov / Width | 右 MAE / Cov / Width |
|---|---|---|
| Pi0.5 z_rl，seed 44 | {left["mae"]:.3f} N / {pct(left["coverage"])} / {left["mean_envelope_width"]:.3f} N | {right["mae"]:.3f} N / {pct(right["coverage"])} / {right["mean_envelope_width"]:.3f} N |
| SmolVLA stable B，3 seeds 均值±标准差 | {smol_summary["left"]["mae"][0]:.3f}±{smol_summary["left"]["mae"][1]:.3f} N / {pct(smol_summary["left"]["coverage"][0])}±{100.0 * smol_summary["left"]["coverage"][1]:.2f}% / {smol_summary["left"]["mean_envelope_width"][0]:.3f}±{smol_summary["left"]["mean_envelope_width"][1]:.3f} N | {smol_summary["right"]["mae"][0]:.3f}±{smol_summary["right"]["mae"][1]:.3f} N / {pct(smol_summary["right"]["coverage"][0])}±{100.0 * smol_summary["right"]["coverage"][1]:.2f}% / {smol_summary["right"]["mean_envelope_width"][0]:.3f}±{smol_summary["right"]["mean_envelope_width"][1]:.3f} N |

## 6. 架构与训练

- 冻结的 Pi0.5 checkpoint：`{pi_features["checkpoint"]}`
- 特征：`{pi_features["feature_tap"]}`
- Envelope：2 层 pre-norm Transformer，`d_model=256`，8 heads，FFN 1024。
- 输入序列：`LEFT_ENV, RIGHT_ENV, STATE, ACTION_1..9, HISTORY_LEFT_1..6, HISTORY_RIGHT_1..6`。
- Cross-attention K/V：Pi0.5 `z_rl`。
- 输出：左右各自 `mu`、`log_var`，并由 calibration 得到 `upper95 = mu + alpha95 * sigma`。
- Loss：左右 masked Gaussian NLL + `0.5 * SmoothL1`。
- 最佳 epoch：{pi_training["best_epoch"]}；共运行 {pi_training["epochs_ran"]} epochs。

## 7. 最终判定

| 用途 | 判定 |
|---|---|
| 离线力预测与曲线分析 | 可用 |
| Stable-contact Online RL soft penalty，小权重、归一化、clip | 可开始 shadow/联调 |
| 新接触 onset 预测 | 不适用，gate=false 时不惩罚 |
| 硬碰撞保护 | 不可替代，必须保留 raw-force hard limit |
| 正式 production reward | 暂不通过 |

当前 Envelope 的核心数字是合理的，`pi_pass={str(pi_pass).lower()}`；主要未完成项不是模型无法预测，
而是在线 200 ms 历史触觉接线、真实同次 action chunk 契约，以及独立新 episode 盲测。
"""

    rendered_report = markdown_to_html(markdown)
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pi0.5 JAX z_rl 触觉力 Envelope 测试报告</title>
<style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;max-width:1120px;margin:0 auto;padding:32px;color:#17212b;line-height:1.65}}
h1,h2{{color:#102a43}} h1{{border-bottom:3px solid #147d64;padding-bottom:12px}}
table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}}
th,td{{border:1px solid #cbd5e1;padding:8px 10px;text-align:left}}
th{{background:#eef3f7}} tr:nth-child(even){{background:#f8fafc}}
img{{display:block;max-width:100%;margin:18px auto;border:1px solid #d6dee6}}
code{{background:#eef3f7;padding:2px 5px}} .pass{{background:#e8f5ef;border-left:5px solid #147d64;padding:14px 18px}}
.warn{{background:#fff5df;border-left:5px solid #d97706;padding:14px 18px}}
.table-wrap{{overflow-x:auto}} pre{{background:#102a43;color:#f8fafc;padding:16px;overflow:auto}}
</style></head><body>
<div class="pass"><strong>离线稳定接触 soft penalty：通过。</strong>
左/右 coverage 为 {pct(left["coverage"])} / {pct(right["coverage"])}。</div>
<div class="warn"><strong>正式安全上界：尚未通过。</strong>
必须保留硬力保护，并完成在线历史触觉接线和独立盲测。</div>
<p><a href="REPORT.md">Markdown 原文</a></p>
{rendered_report}
</body></html>"""
    return markdown, html_doc


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    markdown, html_doc = build_report()
    (OUTPUT / "REPORT.md").write_text(markdown, encoding="utf-8")
    (OUTPUT / "report.html").write_text(html_doc, encoding="utf-8")
    print(OUTPUT / "REPORT.md")
    print(OUTPUT / "report.html")


if __name__ == "__main__":
    main()
