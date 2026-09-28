"""Write Stockfish 19 W/D/L onto the filtered month files.

Each qualifying ply gets sf_w/sf_d/sf_l (per mille, side to move) and has_sf.
The same numbers replace any Lichess %eval inside that move's comment as [%sfwdl w,d,l].
Positions are deduped by piece placement, side to move, castling and en passant.
"""
import argparse, io, os, re, shutil, sqlite3, subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess, chess.pgn, pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

from config import BENCH_NODES, ENGINE, RAW

EVAL = re.compile(r"\[%eval [^]]*\]\s*")
HASH = 1             # smallest table; the sample's worst gap vs Hash 16 was 13 per mille
BATCH = 256          # positions per worker trip; each one is still its own 10k-node search
_ENGINE = None

def check_bench(exe):
    out = subprocess.run([str(exe), "bench"], capture_output=True, text=True, timeout=180)
    text = out.stdout + "\n" + out.stderr
    match = re.search(r"Nodes searched\s*:\s*(\d+)", text)
    got = int(match.group(1)) if match else None
    if got != BENCH_NODES:
        raise SystemExit(f"Stockfish bench searched {got} nodes, expected {BENCH_NODES}. Not labeling.")
    print(f"bench ok: {got} nodes", flush=True)

def _read_until(proc, marker):
    buf = b""
    while marker not in buf:
        chunk = proc.stdout.read(65536)
        if not chunk:
            raise RuntimeError("stockfish exited")
        buf += chunk
    return buf

def _boot(exe):
    proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0, creationflags=0x08000000)
    def send(text):
        proc.stdin.write(text.encode()); proc.stdin.flush()
    proc.send = send
    proc.leftover = b""
    send("uci\n")
    _read_until(proc, b"uciok")
    send(f"setoption name Threads value 1\nsetoption name Hash value {HASH}\n"
         "setoption name UCI_ShowWDL value true\nisready\n")
    _read_until(proc, b"readyok")
    return proc

def _through_bestmove(proc):
    """Read one search, including the whole bestmove line, and keep any extra bytes for the next one."""
    buf = proc.leftover
    while True:
        start = buf.find(b"\nbestmove ")
        if start != -1:
            end = buf.find(b"\n", start + 1)
            if end != -1:
                proc.leftover = buf[end + 1:]
                return buf[:end + 1]
        chunk = proc.stdout.read(65536)
        if not chunk:
            raise RuntimeError("stockfish exited")
        buf += chunk

def _ask(proc, fen):
    """One fresh 10k-node search. ucinewgame finishes before the next command is read."""
    proc.send(f"ucinewgame\nposition fen {fen}\ngo nodes 10000\n")
    wdl = None
    for line in _through_bestmove(proc).splitlines():
        if b" wdl " not in line or b"lowerbound" in line or b"upperbound" in line:
            continue
        parts = line.split()
        i = parts.index(b"wdl")
        wdl = (int(parts[i + 1]), int(parts[i + 2]), int(parts[i + 3]))
    return wdl

def _init(exe):
    global _ENGINE
    _ENGINE = _boot(exe)

def _eval_many(epds):
    return [(epd, _ask(_ENGINE, epd + " 0 1")) for epd in epds]

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

def cache_open(path=None):
    path = Path(path) if path else RAW.parent / "sf_cache.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
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

def parse_games(games):
    return [parse_game(g["movetext"], g["plies"]) for g in games]

def search_missing(pool, con, known, missing, desc=None):
    fresh = []
    if not missing:
        if desc:
            tqdm.write(f"{desc}: cached")
        return
    batches = [missing[i:i + BATCH] for i in range(0, len(missing), BATCH)]
    with tqdm(total=len(missing), desc=desc, unit="pos", unit_scale=True, mininterval=1) as bar:
        for rows in pool.imap_unordered(_eval_many, batches):
            n = 0
            for epd, wdl in rows:
                n += 1
                if wdl is None:
                    continue
                known[epd] = wdl
                fresh.append((epd, *wdl))
                if len(fresh) >= 4000:
                    store(con, fresh); fresh.clear()
            bar.update(n)
    if fresh:
        store(con, fresh)

def columns_for(games, parsed, known):
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
    return movetexts, sf_w, sf_d, sf_l, has_sf

def label_games(games, pool, con, parsed=None):
    parsed = parse_games(games) if parsed is None else parsed
    need = list(dict.fromkeys(epd for epds, _ in parsed for epd in epds.values()))
    known = lookup(con, need)
    missing = [epd for epd in need if epd not in known]
    search_missing(pool, con, known, missing)
    return (*columns_for(games, parsed, known), len(missing))

def prepare_group(path, index, cache_path):
    """Read, parse and cache-lookup one row group. Runs while the previous group is searched."""
    table = pq.ParquetFile(path).read_row_group(index)
    games = table.to_pylist()
    parsed = parse_games(games)
    need = list(dict.fromkeys(epd for epds, _ in parsed for epd in epds.values()))
    reader = sqlite3.connect(cache_path, timeout=120)
    reader.execute("PRAGMA query_only = ON")
    known = lookup(reader, need)
    reader.close()
    missing = [epd for epd in need if epd not in known]
    return table, games, parsed, known, missing

def write_group(table, games, parsed, known, dest):
    cols = columns_for(games, parsed, known)
    pq.write_table(attach(table, *cols[:5]), dest, compression="zstd")

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

def label_file(path, pool, con, games=0, out=None, cache_path=None):
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
    cache_path = str(cache_path or (RAW.parent / "sf_cache.sqlite"))
    parts = path.parent / f"{path.stem}_sfparts"
    parts.mkdir(exist_ok=True)
    n_groups = pq.ParquetFile(path).metadata.num_row_groups
    pending = [(i, parts / f"{i:04d}.parquet") for i in range(n_groups)
               if not (parts / f"{i:04d}.parquet").exists()]
    # The prep thread reads and parses the next group while every engine stays on the current one.
    with ThreadPoolExecutor(max_workers=1) as prep, ThreadPoolExecutor(max_workers=1) as writer:
        nxt = 0
        prep_fut = prep.submit(prepare_group, str(path), pending[0][0], cache_path) if pending else None
        writes = []
        while prep_fut is not None:
            index, dest = pending[nxt]
            table, games, parsed, known, missing = prep_fut.result()
            nxt += 1
            prep_fut = (prep.submit(prepare_group, str(path), pending[nxt][0], cache_path)
                        if nxt < len(pending) else None)
            search_missing(pool, con, known, missing, f"{path.name} {index + 1}/{n_groups}")
            writes.append(writer.submit(write_group, table, games, parsed, known, dest))
        for job in writes:
            job.result()
    done = sorted(parts.glob("*.parquet"))
    if len(done) != n_groups:
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
    ap.add_argument("--workers", type=int, default=16)
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
    with mp.Pool(args.workers, _init, (args.engine,)) as pool:
        if args.games:
            label_file(files[0], pool, con, games=args.games, out=args.out)
            return
        for path in files:
            label_file(path, pool, con)

if __name__ == "__main__":
    main()
