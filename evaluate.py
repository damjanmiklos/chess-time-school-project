"""Log loss, Brier, accuracy and calibration for the test months.

Compares logistic regression, the released Maia-3 value head, and a trained checkpoint.
Breaks each one down by clock ratio, Elo, Stockfish score and time control.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA, RUNS
from data import make_loader, shard_paths
from model import build, context_features, load_pretrained, maia_cfg
from maia3.models import MAIA3Model


class Meter:
    def __init__(self):
        self.n = 0
        self.nll = 0.0
        self.brier = 0.0
        self.correct = 0
        self.count = np.zeros(10)
        self.conf = np.zeros(10)
        self.hit = np.zeros(10)

    def add(self, prob, y):
        prob = np.clip(prob, 1e-7, 1.0)
        n = len(y)
        if n == 0:
            return
        self.n += n
        self.nll += float(-np.log(prob[np.arange(n), y]).sum())
        one = np.zeros_like(prob)
        one[np.arange(n), y] = 1
        self.brier += float(((prob - one) ** 2).sum())
        pred = prob.argmax(1)
        self.correct += int((pred == y).sum())
        conf = prob.max(1)
        bins = np.minimum((conf * 10).astype(int), 9)
        hit = (pred == y).astype(float)
        for i in range(10):
            m = bins == i
            self.count[i] += m.sum()
            self.conf[i] += conf[m].sum()
            self.hit[i] += hit[m].sum()

    def report(self):
        ece = 0.0
        for i in range(10):
            if self.count[i]:
                acc = self.hit[i] / self.count[i]
                avg = self.conf[i] / self.count[i]
                ece += (self.count[i] / self.n) * abs(acc - avg)
        return {"n": self.n, "logloss": self.nll / self.n, "brier": self.brier / self.n,
                "acc": self.correct / self.n, "ece": ece}


def masks(meta):
    cs, co = meta["clk_self"], meta["clk_oppo"]
    ratio = np.maximum(cs, co) / np.maximum(1, np.minimum(cs, co))
    elo, base = meta["self_elo"], meta["base"]
    expect = (meta["sf"][:, 0] + 0.5 * meta["sf"][:, 1]) / 1000
    named = {
        "clock 2-3": (ratio >= 2) & (ratio < 3), "clock 3-5": (ratio >= 3) & (ratio < 5),
        "clock 5-10": (ratio >= 5) & (ratio < 10), "clock 10+": ratio >= 10,
        "elo <1200": elo < 1200, "elo 1200-1600": (elo >= 1200) & (elo < 1600),
        "elo 1600-2000": (elo >= 1600) & (elo < 2000), "elo 2000+": elo >= 2000,
        "sf <0.35": expect < 0.35, "sf 0.35-0.50": (expect >= 0.35) & (expect < 0.50),
        "sf 0.50-0.65": (expect >= 0.50) & (expect < 0.65), "sf >=0.65": expect >= 0.65,
        "tc 600+0": base == 600, "tc longer": base > 600,
    }
    return named.items()


def log_features(meta):
    cs = np.maximum(meta["clk_self"], 1)
    co = np.maximum(meta["clk_oppo"], 1)
    return np.stack([
        meta["self_elo"] - meta["oppo_elo"],
        np.log(cs / co),
        np.log(meta["base"]),
        (meta["sf"][:, 0] + 0.5 * meta["sf"][:, 1]) / 1000,
    ], axis=1)


def numpy_batch(batch):
    return {k: batch[k].numpy() for k in ("clk_self", "clk_oppo", "self_elo", "oppo_elo", "base", "sf", "y")}


def fit_logistic(paths, cfg, limit):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, n = [], [], 0
    for batch in make_loader(paths, cfg, 4096, 0):
        meta = numpy_batch(batch)
        X.append(log_features(meta)); y.append(meta["y"]); n += len(meta["y"])
        if n >= limit:
            break
    X = np.concatenate(X)[:limit]; y = np.concatenate(y)[:limit]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=400).fit(scaler.transform(X), y)
    return clf, scaler


def feed(store, groups, name, prob, y, meta):
    store[name].add(prob, y)
    for key, mask in masks(meta):
        groups[name].setdefault(key, Meter()).add(prob[mask], y[mask])


def show(store, groups):
    names = list(store)
    print(f"{'':16}" + "".join(f"{n:>22}" for n in names))
    header = store[names[0]].report()
    for key in ("n", "logloss", "brier", "acc", "ece"):
        cells = []
        for name in names:
            val = store[name].report()[key]
            cells.append(f"{val:,.0f}" if key == "n" else f"{val:.4f}")
        print(f"{key:16}" + "".join(f"{c:>22}" for c in cells))
    print()
    keys = list(groups[names[0]])
    print(f"{'bucket':16}" + "".join(f"{n + ' logloss':>22}" for n in names))
    for key in keys:
        cells = []
        for name in names:
            m = groups[name].get(key)
            cells.append(f"{m.report()['logloss']:.4f}" if m and m.n else "—")
        print(f"{key:16}" + "".join(f"{c:>22}" for c in cells))
    return header


def plot(store, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="0.7")
    for name, meter in store.items():
        xs, ys = [], []
        for i in range(10):
            if meter.count[i]:
                xs.append(meter.conf[i] / meter.count[i])
                ys.append(meter.hit[i] / meter.count[i])
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path); plt.close()
    print(f"reliability diagram: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test", "both"])
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--logistic-rows", type=int, default=1_000_000)
    args = ap.parse_args()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    paths = [p for s in splits for p in shard_paths(s)]
    if not paths:
        raise SystemExit("no eval shards. Run data.py --split val and --split test.")
    train_paths = shard_paths("train")
    size, where, use_sf = "5m", "both", True
    ckpt = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        size, where, use_sf = ckpt["size"], ckpt["where"], ckpt["use_sf"]
    cfg, spec = maia_cfg(size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maia = MAIA3Model(cfg).to(device).eval()
    load_pretrained(maia, spec)
    model = None
    if ckpt:
        model = build(size, where=where, use_sf=use_sf, checkpoint=args.ckpt).to(device).eval()
    clf, scaler = fit_logistic(train_paths or paths, cfg, args.logistic_rows)
    names = ["logistic", "maia3"] + (["clock"] if model else [])
    store = {n: Meter() for n in names}
    groups = {n: {} for n in names}
    seen = 0
    with torch.no_grad():
        for batch in make_loader(paths, cfg, args.batch, args.workers):
            meta = numpy_batch({k: batch[k].cpu() for k in batch})
            y = meta["y"]
            feed(store, groups, "logistic", clf.predict_proba(scaler.transform(log_features(meta))), y, meta)
            gpu = {k: batch[k].to(device) for k in ("tokens", "self_elo", "oppo_elo", "base", "clk_self", "clk_oppo", "sf")}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, value, _ = maia(gpu["tokens"], gpu["self_elo"], gpu["oppo_elo"])
            feed(store, groups, "maia3", torch.softmax(value.float(), -1).cpu().numpy(), y, meta)
            if model is not None:
                ctx = context_features(gpu["base"], gpu["clk_self"], gpu["clk_oppo"], gpu["sf"])
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    _, value, _ = model(gpu["tokens"], gpu["self_elo"], gpu["oppo_elo"], ctx)
                feed(store, groups, "clock", torch.softmax(value.float(), -1).cpu().numpy(), y, meta)
            seen += len(y)
            if seen >= args.max_rows:
                break
    show(store, groups)
    plot(store, RUNS / "eval" / f"reliability_{args.split}.png")

if __name__ == "__main__":
    main()
