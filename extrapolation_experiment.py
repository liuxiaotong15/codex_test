import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# -----------------------------
# Models (pure Python)
# -----------------------------
class KNNRegressor:
    def __init__(self, k: int = 8):
        self.k = k
        self.x = []
        self.y = []

    def fit(self, x: Sequence[float], y: Sequence[float]):
        self.x = list(x)
        self.y = list(y)

    def predict_one(self, xq: float) -> float:
        pairs = sorted(((abs(xq - xi), yi) for xi, yi in zip(self.x, self.y)), key=lambda z: z[0])
        neigh = pairs[: self.k]
        # distance-weighted average
        num, den = 0.0, 0.0
        for d, yi in neigh:
            w = 1.0 / (d + 1e-6)
            num += w * yi
            den += w
        return num / den

    def predict(self, xs: Sequence[float]) -> List[float]:
        return [self.predict_one(v) for v in xs]


class MLPRegressorSimple:
    """1D -> hidden(tanh) -> 1D MLP."""

    def __init__(self, hidden: int = 32, lr: float = 0.02, epochs: int = 3000):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        # Xavier-like init
        s1 = 1.0 / math.sqrt(1)
        s2 = 1.0 / math.sqrt(hidden)
        self.w1 = [random.uniform(-s1, s1) for _ in range(hidden)]
        self.b1 = [0.0 for _ in range(hidden)]
        self.w2 = [random.uniform(-s2, s2) for _ in range(hidden)]
        self.b2 = 0.0

    def fit(self, x: Sequence[float], y: Sequence[float]):
        n = len(x)
        for _ in range(self.epochs):
            # full-batch gradients
            gw1 = [0.0] * self.hidden
            gb1 = [0.0] * self.hidden
            gw2 = [0.0] * self.hidden
            gb2 = 0.0

            for xi, yi in zip(x, y):
                h = [math.tanh(self.w1[j] * xi + self.b1[j]) for j in range(self.hidden)]
                pred = sum(self.w2[j] * h[j] for j in range(self.hidden)) + self.b2
                dloss_dpred = 2.0 * (pred - yi)

                for j in range(self.hidden):
                    gw2[j] += dloss_dpred * h[j]
                gb2 += dloss_dpred

                for j in range(self.hidden):
                    dh_dz = 1.0 - h[j] * h[j]
                    dloss_dz = dloss_dpred * self.w2[j] * dh_dz
                    gw1[j] += dloss_dz * xi
                    gb1[j] += dloss_dz

            scale = 1.0 / n
            for j in range(self.hidden):
                self.w1[j] -= self.lr * gw1[j] * scale
                self.b1[j] -= self.lr * gb1[j] * scale
                self.w2[j] -= self.lr * gw2[j] * scale
            self.b2 -= self.lr * gb2 * scale

    def predict_one(self, xq: float) -> float:
        h = [math.tanh(self.w1[j] * xq + self.b1[j]) for j in range(self.hidden)]
        return sum(self.w2[j] * h[j] for j in range(self.hidden)) + self.b2

    def predict(self, xs: Sequence[float]) -> List[float]:
        return [self.predict_one(v) for v in xs]


class TreeNode:
    def __init__(self, value: float):
        self.value = value
        self.split = None
        self.left = None
        self.right = None


class DecisionTreeRegressor1D:
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 3):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, x: Sequence[float], y: Sequence[float]):
        pairs = sorted(zip(x, y), key=lambda p: p[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        self.root = self._build(xs, ys, depth=0)

    def _build(self, x: List[float], y: List[float], depth: int) -> TreeNode:
        node = TreeNode(sum(y) / len(y))
        if depth >= self.max_depth or len(x) <= 2 * self.min_samples_leaf:
            return node

        best_score = float("inf")
        best_i = None
        n = len(x)
        for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
            if i >= n:
                continue
            if x[i - 1] == x[i]:
                continue
            left_y = y[:i]
            right_y = y[i:]
            lmean = sum(left_y) / len(left_y)
            rmean = sum(right_y) / len(right_y)
            lerr = sum((v - lmean) ** 2 for v in left_y)
            rerr = sum((v - rmean) ** 2 for v in right_y)
            score = lerr + rerr
            if score < best_score:
                best_score = score
                best_i = i

        if best_i is None:
            return node

        node.split = 0.5 * (x[best_i - 1] + x[best_i])
        node.left = self._build(x[:best_i], y[:best_i], depth + 1)
        node.right = self._build(x[best_i:], y[best_i:], depth + 1)
        return node

    def predict_one(self, xq: float) -> float:
        node = self.root
        while node.split is not None:
            node = node.left if xq <= node.split else node.right
        return node.value

    def predict(self, xs: Sequence[float]) -> List[float]:
        return [self.predict_one(v) for v in xs]


class RandomForestRegressor1D:
    def __init__(self, n_trees: int = 80, max_depth: int = 6, min_samples_leaf: int = 3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, x: Sequence[float], y: Sequence[float]):
        self.trees = []
        n = len(x)
        for _ in range(self.n_trees):
            idx = [random.randrange(n) for _ in range(n)]
            xb = [x[i] for i in idx]
            yb = [y[i] for i in idx]
            tree = DecisionTreeRegressor1D(self.max_depth, self.min_samples_leaf)
            tree.fit(xb, yb)
            self.trees.append(tree)

    def predict(self, xs: Sequence[float]) -> List[float]:
        all_preds = [t.predict(xs) for t in self.trees]
        out = []
        for i in range(len(xs)):
            out.append(sum(p[i] for p in all_preds) / len(all_preds))
        return out


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[], object]
    color: str


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def sample_train_data(n_samples: int) -> Tuple[List[float], List[float]]:
    x = sorted(random.uniform(0.0, 2.0 * math.pi) for _ in range(n_samples))
    y = [math.sin(v) for v in x]
    return x, y


def polyline(points: List[Tuple[float, float]], color: str, width: float = 1.5, opacity: float = 1.0):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" points="{pts}"/>'


def circle(x: float, y: float, r: float, color: str, opacity: float = 1.0):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}" fill-opacity="{opacity}"/>'


def run_experiment(sample_sizes: List[int], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ModelSpec("MLP", lambda: MLPRegressorSimple(hidden=48, lr=0.02, epochs=3500), "#e41a1c"),
        ModelSpec("RandomForest", lambda: RandomForestRegressor1D(n_trees=100, max_depth=6, min_samples_leaf=3), "#377eb8"),
        ModelSpec("KNN", lambda: KNNRegressor(k=8), "#4daf4a"),
    ]

    x_eval = [4.0 * math.pi * i / 900 for i in range(901)]
    y_true = [math.sin(v) for v in x_eval]
    ex_idx = [i for i, xv in enumerate(x_eval) if xv >= 2.0 * math.pi]

    metrics = {m.name: {} for m in models}

    # SVG canvas setup
    w, h = 1200, 340 * len(sample_sizes)
    margin_l, margin_r, margin_t, margin_b = 80, 40, 30, 40
    panel_h = (h - margin_t - margin_b) / len(sample_sizes)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    for row, n_samples in enumerate(sample_sizes):
        x_train, y_train = sample_train_data(n_samples)
        top = margin_t + row * panel_h
        bottom = top + panel_h - 20
        left, right = margin_l, w - margin_r

        def tx(xv: float) -> float:
            return left + (xv / (4.0 * math.pi)) * (right - left)

        def ty(yv: float) -> float:
            return bottom - ((yv + 1.3) / 2.6) * (bottom - top)

        # axes
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444" stroke-width="1"/>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444" stroke-width="1"/>')

        # split line at 2pi
        x_split = tx(2.0 * math.pi)
        svg.append(f'<line x1="{x_split:.2f}" y1="{top}" x2="{x_split:.2f}" y2="{bottom}" stroke="#999" stroke-dasharray="5,5"/>')
        svg.append(f'<text x="{x_split + 6:.2f}" y="{top + 14:.2f}" fill="#666" font-size="12">extrapolation</text>')

        # truth
        true_pts = [(tx(xv), ty(yv)) for xv, yv in zip(x_eval, y_true)]
        svg.append(polyline(true_pts, "#000000", width=2.0))

        # training points
        for xv, yv in zip(x_train, y_train):
            svg.append(circle(tx(xv), ty(yv), 2.5, "#222", 0.8))

        # model lines + metrics
        legend_y = top + 22
        svg.append(f'<text x="{left}" y="{top - 8:.2f}" fill="#111" font-size="14">Training size n={n_samples}</text>')
        for i, model_spec in enumerate(models):
            model = model_spec.builder()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_eval)
            ext_true = [y_true[j] for j in ex_idx]
            ext_pred = [y_pred[j] for j in ex_idx]
            e = rmse(ext_true, ext_pred)
            metrics[model_spec.name][n_samples] = e

            pred_pts = [(tx(xv), ty(yv)) for xv, yv in zip(x_eval, y_pred)]
            svg.append(polyline(pred_pts, model_spec.color, width=1.7, opacity=0.9))

            ly = legend_y + i * 16
            svg.append(f'<line x1="{right - 360}" y1="{ly - 4}" x2="{right - 340}" y2="{ly - 4}" stroke="{model_spec.color}" stroke-width="2"/>')
            svg.append(
                f'<text x="{right - 335}" y="{ly:.2f}" fill="#222" font-size="12">{model_spec.name}  RMSE[2π,4π]={e:.3f}</text>'
            )

    svg.append("</svg>")
    svg_text = "\n".join(svg)

    plot_path = output_dir / "sin_extrapolation_comparison.svg"
    plot_path.write_text(svg_text, encoding="utf-8")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *sample_sizes])
        for model_name, row in metrics.items():
            writer.writerow([model_name, *[f"{row[n]:.6f}" for n in sample_sizes]])

    print(f"Saved plot to: {plot_path}")
    print(f"Saved metrics to: {metrics_path}")
    for model_name, row in metrics.items():
        s = ", ".join(f"n={n}: {row[n]:.4f}" for n in sample_sizes)
        print(f"{model_name}: {s}")


if __name__ == "__main__":
    run_experiment(sample_sizes=[20, 50, 120], output_dir=Path("results"))
