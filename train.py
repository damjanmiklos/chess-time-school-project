"""Fine-tune ClockMaia3 for one pass. Value loss plus a small policy loss.

Warm-up (5% of steps): only the new columns, context MLP and value head.
Then everything unfreezes. Pretrained weights use 10x lower AdamW learning rate.
"""
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import RUNS
from data import make_loader, shard_paths
from model import build, context_features, maia_cfg, split_params


def save(model, run, step, meta):
    payload = {"model": model.state_dict(), "step": step, **meta}
    torch.save(payload, run / "last.pt")
    if step % 2000 == 0:
        torch.save(payload, run / f"step{step:07d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="5m", choices=["5m", "23m"])
    ap.add_argument("--where", default="both", choices=["both", "head", "film"])
    ap.add_argument("--no-sf", action="store_true")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="train")
    ap.add_argument("--data", default=None, help="shard folder; default data/<split>")
    ap.add_argument("--steps", type=int, default=0, help="stop early; 0 means one full pass")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    torch.backends.cuda.matmul.allow_tf32 = True
    paths = shard_paths(args.split, args.data)
    if not paths:
        raise SystemExit(f"no shards for {args.split}. Run data.py after label_sf.py.")
    import pyarrow.parquet as pq
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    total = args.steps or max(1, -(-rows // args.batch))
    warmup = int(round(0.05 * total))
    name = args.name or (f"{args.size}-{args.where}" + ("-nosf" if args.no_sf else ""))
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    print(f"{rows:,} positions, {total} steps, warmup {warmup}, {torch.cuda.get_device_name(0)}", flush=True)

    cfg, _ = maia_cfg(args.size)
    model = build(args.size, where=args.where, use_sf=not args.no_sf).cuda().train()
    high, low = split_params(model)
    meta = {"size": args.size, "where": args.where, "use_sf": not args.no_sf}

    def make_opt(step):
        main_phase = step >= warmup
        for p in low:
            p.requires_grad_(main_phase)
        for p in high:
            p.requires_grad_(True)
        if not main_phase:
            return torch.optim.AdamW(high, lr=args.lr, weight_decay=0.01)
        return torch.optim.AdamW([
            {"params": low, "lr": args.lr / 10},
            {"params": high, "lr": args.lr},
        ], weight_decay=0.01)

    opt = make_opt(0)
    loader = make_loader(paths, cfg, args.batch, args.workers)
    seen, t0, log = 0, time.time(), open(run / "log.jsonl", "a", encoding="utf-8")
    for step, batch in enumerate(loader):
        if step == warmup and warmup > 0:
            opt = make_opt(step)
        if step >= total:
            break
        if step < warmup:
            opt.param_groups[0]["lr"] = args.lr * (step + 1) / max(1, warmup)
        else:
            scale = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            if len(opt.param_groups) == 1:
                opt.param_groups[0]["lr"] = args.lr * scale
            else:
                opt.param_groups[0]["lr"] = (args.lr / 10) * scale
                opt.param_groups[1]["lr"] = args.lr * scale
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        ctx = context_features(batch["base"], batch["clk_self"], batch["clk_oppo"], batch["sf"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_move, logits_value, _ = model(batch["tokens"], batch["self_elo"], batch["oppo_elo"], ctx)
        loss = (F.cross_entropy(logits_value.float(), batch["y"])
                + args.policy * F.cross_entropy(logits_move.float(), batch["move"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += batch["y"].shape[0]
        if step % 20 == 0 or step + 1 == total:
            rate = seen / max(1e-6, time.time() - t0)
            row = {"step": step, "loss": loss.item(), "pos_per_s": rate, "per_day": rate * 86400}
            print(f"step {step:6d}  loss {loss.item():.4f}  {rate:,.0f} pos/s  {rate * 86400 / 1e9:.2f} B/day", flush=True)
            log.write(json.dumps(row) + "\n"); log.flush()
        if step % 2000 == 0:
            save(model, run, step, meta)
    save(model, run, step, meta)
    log.close()
    print(f"saved {run / 'last.pt'}", flush=True)

if __name__ == "__main__":
    main()
