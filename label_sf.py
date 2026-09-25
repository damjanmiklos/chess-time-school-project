"""Write Stockfish 19 W/D/L onto the filtered month files.

Each qualifying ply gets sf_w/sf_d/sf_l (per mille, side to move) and has_sf.
The same numbers replace any Lichess %eval inside that move's comment as [%sfwdl w,d,l].
Positions are deduped by piece placement, side to move, castling and en passant.
"""
import argparse, io, os, re, shutil, sqlite3, subprocess
from multiprocessing import Pool
from pathlib import Path

import chess, chess.pgn, pyarrow as pa, pyarrow.parquet as pq

from config import BENCH_NODES, ENGINE, RAW

EVAL = re.compile(r"\[%eval [^]]*\]\s*")
_ENGINE = None

def check_bench(exe):
    out = subprocess.run([str(exe), "bench"], capture_output=True, text=True, timeout=180)
    text = out.stdout + "\n" + out.stderr
    match = re.search(r"Nodes searched\s*:\s*(\d+)", text)
    got = int(match.group(1)) if match else None
    if got != BENCH_NODES:
        raise SystemExit(f"Stockfish bench searched {got} nodes, expected {BENCH_NODES}. Not labeling.")
    print(f"bench ok: {got} nodes", flush=True)

def _boot(exe):
    proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(line):
        proc.stdin.write(line + "\n"); proc.stdin.flush()
    def readline():
        return proc.stdout.readline()
    send("uci")
    while not readline().startswith("uciok"):
        pass
    send("setoption name Threads value 1")
    send("setoption name Hash value 16")
    send("setoption name UCI_ShowWDL value true")
    send("isready")
    while readline().strip() != "readyok":
        pass
    proc.send, proc.readline = send, readline
    return proc

def _ask(proc, fen):
    proc.send("ucinewgame"); proc.send("isready")
    while proc.readline().strip() != "readyok":
        pass
    proc.send(f"position fen {fen}")
    proc.send("go nodes 10000")
    wdl = None
    while True:
        line = proc.readline()
        if not line:
            raise RuntimeError("stockfish exited")
        if " wdl " in line and "lowerbound" not in line and "upperbound" not in line:
            parts = line.split(); i = parts.index("wdl")
            wdl = (int(parts[i + 1]), int(parts[i + 2]), int(parts[i + 3]))
        if line.startswith("bestmove"):
            return wdl

def _init(exe):
    global _ENGINE
    _ENGINE = _boot(exe)

def _eval(epd):
    return epd, _ask(_ENGINE, epd + " 0 1")

def parse_game(movetext, plies):
    game = chess.pgn.read_game(io.StringIO('[Result "*"]\n\n' + movetext))
    if game is None:
        return {}, 0
    board, want, out, n = game.board(), {int(p) for p in plies}, {}, 0
    for i, node in enumerate(game.mainline()):
        if i in want:
            out[i] = board.epd()          # pieces, side, castling, en passant — no move clocks
        board.push(node.move)
        n = i + 1
    return out, n

def stamp(movetext, ply_wdl, n_moves):
    comments = re.findall(r"\{[^}]*\}", movetext)
    if len(comments) != n_moves:
        return EVAL.sub("", movetext)
    i = 0
    def repl(match):
        nonlocal i
        body = EVAL.sub("", match.group(0))
        wdl = ply_wdl.get(i)
        i += 1
        if wdl is None:
            return body
        return body[:-1] + f" [%sfwdl {wdl[0]},{wdl[1]},{wdl[2]}]" + "}"
    return re.sub(r"\{[^}]*\}", repl, movetext)

def cache_open():
    path = RAW.parent / "sf_cache.sqlite"
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS pos (epd TEXT PRIMARY KEY, w INT, d INT, l INT)")
    return con

def lookup(con, epds):
    found = {}
    for start in range(0, len(epds), 400):
        chunk = epds[start:start + 400]
        q = ",".join("?" * len(chunk))
        for epd, w, d, l in con.execute(f"SELECT epd, w, d, l FROM pos WHERE epd IN ({q})", chunk):
            found[epd] = (w, d, l)
    return found

def store(con, rows):
    con.executemany("INSERT OR IGNORE INTO pos VALUES (?,?,?,?)", rows)
    con.commit()

def label_games(games, pool, con):
    parsed = [parse_game(g["movetext"], g["plies"]) for g in games]
    need = list(dict.fromkeys(epd for epds, _ in parsed for epd in epds.values()))
    known = lookup(con, need)
    missing = [epd for epd in need if epd not in known]
    fresh, done = [], 0
    if missing:
        print(f"  evaluating {len(missing):,} new positions ({len(known):,} cached)", flush=True)
        for epd, wdl in pool.imap_unordered(_eval, missing, chunksize=16):
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(missing):,}", flush=True)
            if wdl is None:
                continue
            known[epd] = wdl
            fresh.append((epd, *wdl))
            if len(fresh) >= 4000:
                store(con, fresh); fresh.clear()
        if fresh:
            store(con, fresh)
    movetexts, sf_w, sf_d, sf_l, has_sf = [], [], [], [], []
    for game, (epds, n_moves) in zip(games, parsed):
        ww, dd, ll, flags, ply_wdl = [], [], [], [], {}
        for ply in game["plies"]:
            wdl = known.get(epds.get(int(ply)))
            if wdl is None:
                ww.append(0); dd.append(0); ll.append(0); flags.append(False)
            else:
                ww.append(wdl[0]); dd.append(wdl[1]); ll.append(wdl[2]); flags.append(True)
                ply_wdl[int(ply)] = wdl
        movetexts.append(stamp(game["movetext"], ply_wdl, n_moves))
        sf_w.append(ww); sf_d.append(dd); sf_l.append(ll); has_sf.append(flags)
    return movetexts, sf_w, sf_d, sf_l, has_sf, len(missing)

def attach(table, movetexts, sf_w, sf_d, sf_l, has_sf):
    keep = [n for n in table.schema.names if n not in ("sf_w", "sf_d", "sf_l", "has_sf")]
    table = table.select(keep)
    i = table.schema.get_field_index("movetext")
    table = table.set_column(i, "movetext", pa.array(movetexts))
    table = table.append_column("sf_w", pa.array(sf_w, pa.list_(pa.uint16())))
    table = table.append_column("sf_d", pa.array(sf_d, pa.list_(pa.uint16())))
    table = table.append_column("sf_l", pa.array(sf_l, pa.list_(pa.uint16())))
    table = table.append_column("has_sf", pa.array(has_sf, pa.list_(pa.bool_())))
    return table

def label_file(path, pool, con, games=0, out=None):
    path = Path(path)
    if games:
        table = pq.ParquetFile(path).read_row_group(0).slice(0, games)
        cols = label_games(table.to_pylist(), pool, con)
        labeled = attach(table, *cols[:5])
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(labeled, dest, compression="zstd")
        print(f"wrote {games} games to {dest}, new positions {cols[5]}", flush=True)
        return
    if "has_sf" in pq.read_schema(path).names:
        print(f"skip {path.name}: already labeled", flush=True)
        return
    parts = path.parent / f"{path.stem}_sfparts"
    parts.mkdir(exist_ok=True)
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        dest = parts / f"{i:04d}.parquet"
        if dest.exists():
            continue
        table = pf.read_row_group(i)
        cols = label_games(table.to_pylist(), pool, con)
        pq.write_table(attach(table, *cols[:5]), dest, compression="zstd")
        print(f"{path.name} group {i + 1}/{pf.num_row_groups}: {table.num_rows} games, "
              f"{cols[5]} new positions", flush=True)
    done = sorted(parts.glob("*.parquet"))
    if len(done) != pf.num_row_groups:
        raise SystemExit(f"{path.name} incomplete ({len(done)}/{pf.num_row_groups})")
    tmp = path.with_suffix(".parquet.stitch")
    writer = None
    for part in done:
        table = pq.read_table(part)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
    writer.close()
    os.replace(tmp, path)
    shutil.rmtree(parts)
    print(f"replaced {path.name}", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Add Stockfish W/D/L columns to the filtered months.")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--month", default=None, help="only this YYYY-MM")
    ap.add_argument("--games", type=int, default=0, help="label this many games into --out and stop")
    ap.add_argument("--out", default="scratch/sf_sample.parquet")
    ap.add_argument("--engine", default=str(ENGINE))
    args = ap.parse_args()
    if not Path(args.engine).exists():
        raise SystemExit(f"missing engine {args.engine}")
    check_bench(args.engine)
    files = sorted(RAW.glob("lichess_*_filtered.parquet"))
    if args.month:
        files = [p for p in files if f"_{args.month}_" in p.name]
    if not files:
        raise SystemExit("no month files")
    con = cache_open()
    with Pool(args.workers, _init, (args.engine,)) as pool:
        if args.games:
            label_file(files[0], pool, con, games=args.games, out=args.out)
            return
        for path in files:
            label_file(path, pool, con)

if __name__ == "__main__":
    main()
