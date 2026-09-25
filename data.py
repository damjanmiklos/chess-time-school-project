"""Sample up to 5 positions per game, bucket-shuffle them, and build Maia tensors in the workers.

Label y is the game result from the side to move: 0 loss, 1 draw, 2 win.
Even plies are White to move. self/oppo elo and clock follow that side.
"""
import argparse, io, re, zlib
from collections import deque
from pathlib import Path

import chess, chess.pgn, numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from config import DATA, month_files
from maia3.dataset import get_historical_tokens, tokenize_board
from maia3.utils import get_all_possible_moves, mirror_move

CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+)\]")
MOVE_INDEX = {m: i for i, m in enumerate(get_all_possible_moves())}
SCHEMA = pa.schema([
    ("moves", pa.string()), ("ply", pa.int16()),
    ("self_elo", pa.int16()), ("oppo_elo", pa.int16()), ("y", pa.int8()),
    ("base", pa.int32()), ("clk_self", pa.int32()), ("clk_oppo", pa.int32()),
    ("sf_w", pa.uint16()), ("sf_d", pa.uint16()), ("sf_l", pa.uint16()),
])

def base_seconds(tc):
    return int(str(tc).split("+")[0])

def clocks_before(movetext, base):
    """Clock pair (white, black) before each ply. %clk is the mover's time after the move."""
    clks = [int(a) * 3600 + int(b) * 60 + int(c) for a, b, c in CLK.findall(movetext)]
    w = b = base
    out = []
    for ply, c in enumerate(clks):
        out.append((w, b))
        if ply % 2 == 0: w = c
        else:            b = c
    return out

def uci_moves(movetext):
    game = chess.pgn.read_game(io.StringIO('[Result "*"]\n\n' + movetext))
    if game is None:
        return []
    return [node.move.uci() for node in game.mainline()]

def outcome(result, white_to_move):
    if result == "1/2-1/2":
        return 1
    if result not in ("1-0", "0-1"):
        return None
    return 2 if (result == "1-0") == white_to_move else 0

def encode_row(moves, ply, self_elo, oppo_elo, y, base, clk_self, clk_oppo, sf, cfg):
    board = chess.Board()
    hist = deque([tokenize_board(board)], maxlen=cfg.history)
    for i, uci in enumerate(moves):
        if i == ply:
            break
        board.push_uci(uci)
        hist.append(tokenize_board(board))
    if ply >= len(moves):
        return None
    uci = moves[ply]
    if board.turn == chess.BLACK:
        uci = mirror_move(uci)
    if uci not in MOVE_INDEX:
        return None
    return {
        "tokens": get_historical_tokens(hist, cfg, 0, 0, 0, 0),
        "self_elo": torch.tensor(int(self_elo)), "oppo_elo": torch.tensor(int(oppo_elo)),
        "base": torch.tensor(float(base)), "clk_self": torch.tensor(float(clk_self)),
        "clk_oppo": torch.tensor(float(clk_oppo)), "sf": torch.tensor(sf, dtype=torch.float32),
        "y": torch.tensor(int(y)), "move": torch.tensor(MOVE_INDEX[uci]),
    }

def encode_game(game, slots, cfg, limit=None):
    """Turn selected indexes into game['plies'] into model rows. Missing Stockfish becomes zeros."""
    moves = uci_moves(game["movetext"])
    clocks = clocks_before(game["movetext"], base_seconds(game["TimeControl"]))
    has, sfw, sfd, sfl = game.get("has_sf"), game.get("sf_w"), game.get("sf_d"), game.get("sf_l")
    rows = []
    for slot in slots:
        ply = int(game["plies"][slot])
        if ply >= len(moves) or ply >= len(clocks):
            continue
        if has is not None and not has[slot]:
            continue
        white = ply % 2 == 0
        y = outcome(game["Result"], white)
        if y is None:
            continue
        wclk, bclk = clocks[ply]
        sf = [0, 0, 0] if sfw is None else [int(sfw[slot]), int(sfd[slot]), int(sfl[slot])]
        row = encode_row(
            moves, ply,
            game["WhiteElo"] if white else game["BlackElo"],
            game["BlackElo"] if white else game["WhiteElo"],
            y, base_seconds(game["TimeControl"]),
            wclk if white else bclk, bclk if white else wclk, sf, cfg)
        if row:
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows

def sample_rows(game, rng, k=5):
    """At most k labeled qualifying plies. Returns compact shard rows, not tensors."""
    has = game.get("has_sf")
    if not has:
        return []
    slots = [i for i, ok in enumerate(has) if ok]
    if not slots:
        return []
    take = slots if len(slots) <= k else list(rng.choice(np.array(slots), size=k, replace=False))
    moves = uci_moves(game["movetext"])
    base = base_seconds(game["TimeControl"])
    clocks = clocks_before(game["movetext"], base)
    rows = []
    for slot in take:
        ply = int(game["plies"][slot])
        if ply >= len(moves) or ply >= len(clocks):
            continue
        white = ply % 2 == 0
        y = outcome(game["Result"], white)
        if y is None:
            continue
        wclk, bclk = clocks[ply]
        rows.append({
            "moves": " ".join(moves[:ply + 1]), "ply": ply,
            "self_elo": int(game["WhiteElo"] if white else game["BlackElo"]),
            "oppo_elo": int(game["BlackElo"] if white else game["WhiteElo"]),
            "y": y, "base": base,
            "clk_self": int(wclk if white else bclk), "clk_oppo": int(bclk if white else wclk),
            "sf_w": int(game["sf_w"][slot]), "sf_d": int(game["sf_d"][slot]), "sf_l": int(game["sf_l"][slot]),
        })
    return rows

def _flush(bufs, writers, folder, i):
    if not bufs[i]["moves"]:
        return
    if i not in writers:
        writers[i] = pq.ParquetWriter(folder / f"{i:04d}.parquet", SCHEMA, compression="zstd")
    writers[i].write_table(pa.table(bufs[i], schema=SCHEMA))
    for col in bufs[i].values():
        col.clear()

def write_buckets(files, folder, buckets, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    bufs = {i: {name: [] for name in SCHEMA.names} for i in range(buckets)}
    writers, rng, n = {}, np.random.default_rng(seed), 0
    for path in files:
        if "has_sf" not in pq.read_schema(path).names:
            raise SystemExit(f"{path.name} has no Stockfish labels yet. Run label_sf.py first.")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
            for game in batch.to_pylist():
                grow = np.random.default_rng(zlib.crc32(str(game["Site"]).encode()) ^ seed)
                for row in sample_rows(game, grow):
                    b = int(rng.integers(buckets))
                    for key, val in row.items():
                        bufs[b][key].append(val)
                    if len(bufs[b]["moves"]) >= 2000:
                        _flush(bufs, writers, folder, b)
                    n += 1
        print(f"{Path(path).name}: {n:,} samples", flush=True)
    for i in range(buckets):
        _flush(bufs, writers, folder, i)
    for w in writers.values():
        w.close()
    return n

def shuffle_buckets(folder, out, seed=1):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for path in sorted(folder.glob("*.parquet")):
        table = pq.read_table(path)
        order = rng.permutation(table.num_rows)
        pq.write_table(table.take(order), out / path.name, compression="zstd")
        path.unlink()
    folder.rmdir()

class ShardDataset(IterableDataset):
    def __init__(self, paths, cfg):
        self.paths = [str(p) for p in paths]
        self.cfg = cfg

    def __iter__(self):
        paths = self.paths
        info = get_worker_info()
        if info is not None:
            paths = paths[info.id::info.num_workers]
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    item = encode_row(
                        row["moves"].split(), int(row["ply"]), row["self_elo"], row["oppo_elo"],
                        row["y"], row["base"], row["clk_self"], row["clk_oppo"],
                        [row["sf_w"], row["sf_d"], row["sf_l"]], self.cfg)
                    if item:
                        yield item

def collate(rows):
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}

def shard_paths(split=None, folder=None):
    folder = Path(folder) if folder else DATA / split
    return sorted(folder.glob("*.parquet"))

def make_loader(paths, cfg, batch, workers):
    ds = ShardDataset(paths, cfg)
    return DataLoader(ds, batch_size=batch, collate_fn=collate, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)

def main():
    ap = argparse.ArgumentParser(description="Sample 5 positions per game and bucket-shuffle them.")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--buckets", type=int, default=1000)
    args = ap.parse_args()
    files = [Path(p) for p in args.files] if args.files else month_files(args.split)
    out = Path(args.out) if args.out else DATA / args.split
    buckets = Path(str(out) + "_buckets")
    n = write_buckets(files, buckets, args.buckets)
    shuffle_buckets(buckets, out)
    print(f"wrote {n:,} shuffled samples to {out}")

if __name__ == "__main__":
    main()
