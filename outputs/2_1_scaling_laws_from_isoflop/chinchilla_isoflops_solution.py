"""Solve the CS336 chinchilla_isoflops problem.

This script implements the IsoFLOPs procedure from the handout:
1. group runs by compute budget,
2. choose the run with the lowest final loss for each budget,
3. compute D_opt = C / (6 N_opt),
4. fit log-log power laws for N_opt(C) and D_opt(C),
5. report predictions for 1e23 and 1e24 FLOPs.

It uses only the Python standard library and writes SVG plots.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "isoflops_curves.json"
OUTPUT_DIR = ROOT / "outputs"


def fit_power_law(points: list[dict[str, float]], key: str) -> tuple[float, float]:
    """Fit y = coefficient * C ** exponent by least squares in log space."""
    xs = [math.log(p["C"]) for p in points]
    ys = [math.log(p[key]) for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    exponent = numerator / denominator
    log_coefficient = y_mean - exponent * x_mean
    return math.exp(log_coefficient), exponent


def predict(coefficient: float, exponent: float, compute_budget: float) -> float:
    return coefficient * compute_budget**exponent


def sci(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}".replace("e+", "e")


def full(x: float) -> str:
    return f"{round(x):,}"


def make_svg(
    points: list[dict[str, float]],
    coefficient: float,
    exponent: float,
    key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 92, 28, 52, 78

    x_min = math.log10(min(p["C"] for p in points) / 1.2)
    x_max = 24.0
    line_cs = [10 ** (x_min + i * (x_max - x_min) / 159) for i in range(160)]
    y_values = [p[key] for p in points] + [predict(coefficient, exponent, c) for c in line_cs]
    y_min = math.log10(min(y_values) / 1.25)
    y_max = math.log10(max(y_values) * 1.25)

    def px(compute_budget: float) -> float:
        return left + (math.log10(compute_budget) - x_min) / (x_max - x_min) * (width - left - right)

    def py(value: float) -> float:
        return height - bottom - (math.log10(value) - y_min) / (y_max - y_min) * (height - top - bottom)

    line = " ".join(
        f"{'M' if i == 0 else 'L'} {px(c):.2f} {py(predict(coefficient, exponent, c)):.2f}"
        for i, c in enumerate(line_cs)
    )
    x_ticks = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
    y_ticks = [10**e for e in range(8, 15) if y_min <= e <= y_max]

    x_grid = "\n  ".join(
        f'<line x1="{px(t):.2f}" y1="{top}" x2="{px(t):.2f}" y2="{height - bottom}" stroke="#ddd" />'
        f'<text x="{px(t):.2f}" y="{height - bottom + 28}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="13" fill="#3f3f46">1e{round(math.log10(t))}</text>'
        for t in x_ticks
    )
    y_grid = "\n  ".join(
        f'<line x1="{left}" y1="{py(t):.2f}" x2="{width - right}" y2="{py(t):.2f}" stroke="#e5e2dc" />'
        f'<text x="{left - 12}" y="{py(t):.2f}" dy="4" text-anchor="end" '
        f'font-family="Arial, sans-serif" font-size="13" fill="#3f3f46">1e{round(math.log10(t))}</text>'
        for t in y_ticks
    )
    circles = "\n  ".join(
        f'<circle cx="{px(p["C"]):.2f}" cy="{py(p[key]):.2f}" r="5.5" fill="#d95f02" '
        f'stroke="#fff" stroke-width="1.5"><title>C={sci(p["C"])}, {key}={sci(p[key])}</title></circle>'
        for p in points
    )

    output_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7"/>
  <text x="{left}" y="30" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2933">{title}</text>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.4"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.4"/>
  {x_grid}
  {y_grid}
  <path d="{line}" fill="none" stroke="#116466" stroke-width="3"/>
  {circles}
  <text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#1f2933">Compute budget C (FLOPs, log scale)</text>
  <text transform="translate(25 {height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#1f2933">{ylabel} (log scale)</text>
  <rect x="{width - 328}" y="58" width="292" height="56" rx="6" fill="#fff" stroke="#d8d5cf"/>
  <line x1="{width - 306}" y1="78" x2="{width - 260}" y2="78" stroke="#116466" stroke-width="3"/>
  <circle cx="{width - 283}" cy="96" r="5.5" fill="#d95f02"/>
  <text x="{width - 246}" y="83" font-family="Arial, sans-serif" font-size="13" fill="#1f2933">fitted power law</text>
  <text x="{width - 246}" y="101" font-family="Arial, sans-serif" font-size="13" fill="#1f2933">IsoFLOPs optima</text>
</svg>
""",
        encoding="utf-8",
    )


def main() -> None:
    runs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    groups: dict[float, list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        groups[float(run["compute_budget"])].append(run)

    optima = []
    for compute_budget in sorted(groups):
        best = min(groups[compute_budget], key=lambda run: run["final_loss"])
        n_opt = float(best["parameters"])
        optima.append(
            {
                "C": compute_budget,
                "N": n_opt,
                "D": compute_budget / (6.0 * n_opt),
                "loss": float(best["final_loss"]),
            }
        )

    n_coefficient, n_exponent = fit_power_law(optima, "N")
    d_coefficient, d_exponent = fit_power_law(optima, "D")

    make_svg(
        optima,
        n_coefficient,
        n_exponent,
        "N",
        "IsoFLOPs Scaling Law: Compute-Optimal Model Size",
        "Optimal model size N",
        OUTPUT_DIR / "chinchilla_isoflops_model_size.svg",
    )
    make_svg(
        optima,
        d_coefficient,
        d_exponent,
        "D",
        "IsoFLOPs Scaling Law: Compute-Optimal Dataset Size",
        "Optimal dataset size D",
        OUTPUT_DIR / "chinchilla_isoflops_dataset_size.svg",
    )

    print("IsoFLOPs optima used for fitting:")
    for p in optima:
        print(f"C={sci(p['C'])}, N_opt={full(p['N'])}, D_opt={full(p['D'])}, loss={p['loss']:.6f}")
    print(f"\nN_opt(C) = {sci(n_coefficient)} * C^{n_exponent:.6f}")
    print(f"D_opt(C) = {sci(d_coefficient)} * C^{d_exponent:.6f}")
    for compute_budget in (1e23, 1e24):
        print(
            f"C={sci(compute_budget, 0)}: "
            f"N_opt={full(predict(n_coefficient, n_exponent, compute_budget))}, "
            f"D_opt={full(predict(d_coefficient, d_exponent, compute_budget))}"
        )


if __name__ == "__main__":
    main()
