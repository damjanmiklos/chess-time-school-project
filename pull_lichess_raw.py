"""Build a filtered clock-imbalance dataset from the raw Lichess monthly dumps.

For each month: download lichess_db_standard_rated_YYYY-MM.pgn.zst (resumable), stream-parse it,
keep games that pass the filters, write one parquet file per month, delete the raw dump.

Filters (header): rated, no BOTs, not abandoned, base >= 600 s, increment 0, |Elo diff| <= 150, has %clk
Filter (moves):   game kept if >= 1 position where one clock >= 2x the other.
Output columns:   Site, UTCDate, WhiteElo, BlackElo, Result, Termination, TimeControl, movetext,
                  plies (index of every qualifying position = the ply about to be played; even = White to move)

Usage:  python pull_lichess_raw.py --from 2025-08 --workers 2 --out C:/dev/chessdata
"""
import argparse, datetime as dt, io, os, re, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa, pyarrow.parquet as pq, requests, zstandard

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+)\]")
TAG = re.compile(r'\[(\w+) "(.*)"\]')
RATIO, MIN_BASE, MAX_ELO_DIFF = 2.0, 600, 150
BATCH = 20_000

SCHEMA = pa.schema([("Site", pa.string()), ("UTCDate", pa.date32()), ("WhiteElo", pa.int16()),
                    ("BlackElo", pa.int16()), ("Result", pa.string()), ("Termination", pa.string()),
                    ("TimeControl", pa.string()), ("movetext", pa.string()), ("plies", pa.list_(pa.int16()))])

# ---------------------------------------------------------------- filters
def header_ok(h):
    """Cheap checks on the PGN headers; returns base time in seconds or None."""
    if not h.get("Event", "").startswith("Rated"): return None
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT": return None
    if h.get("Termination") == "Abandoned": return None
    tc = h.get("TimeControl", "-")
    if not tc.endswith("+0"): return None
    try:
        base = int(tc.split("+")[0]); we, be = int(h["WhiteElo"]), int(h["BlackElo"])
    except (ValueError, KeyError):
        return None
    if base < MIN_BASE or abs(we - be) > MAX_ELO_DIFF: return None
    return base

def qualifying_plies(movetext, base):
    """Clocks (white, black) before each ply; ply i is played by White if i is even."""
    clks = [int(a)*3600 + int(b)*60 + int(c) for a, b, c in CLK.findall(movetext)]
    w = b = base
    out = []
    for ply, c in enumerate(clks):
        hi, lo = (w, b) if w >= b else (b, w)
        if hi >= RATIO * lo:
            out.append(ply)
        if ply % 2 == 0: w = c
        else:            b = c
    return out

# ---------------------------------------------------------------- parsing
def iter_games(path):
    """Yield (headers, movetext) from a .pgn.zst file, streaming."""
    with open(path, "rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f, read_size=1 << 22),
                                encoding="utf-8", errors="replace")
        headers, moves = {}, []
        for line in text:
            if line.startswith("["):
                if moves:
                    yield headers, " ".join(moves)
                    headers, moves = {}, []
                m = TAG.match(line)
                if m: headers[m.group(1)] = m.group(2)
            elif line.strip():
                moves.append(line.strip())
        if moves:
            yield headers, " ".join(moves)

def filter_file(src, dst, log=lambda *a: None):
    tmp = dst + ".tmp"
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    cols = {k: [] for k in SCHEMA.names}
    n = passed = kept = npos = 0
    def flush():
        if cols["Site"]:
            writer.write_table(pa.table(cols, schema=SCHEMA))
            for v in cols.values(): v.clear()
    try:
        for h, mt in iter_games(src):
            n += 1
            base = header_ok(h)
            if base is None or "%clk" not in mt: continue
            passed += 1
            plies = qualifying_plies(mt, base)
            if not plies: continue
            kept += 1; npos += len(plies)
            cols["Site"].append(h.get("Site")); cols["UTCDate"].append(dt.datetime.strptime(h["UTCDate"], "%Y.%m.%d").date())
            cols["WhiteElo"].append(int(h["WhiteElo"])); cols["BlackElo"].append(int(h["BlackElo"]))
            cols["Result"].append(h.get("Result")); cols["Termination"].append(h.get("Termination"))
            cols["TimeControl"].append(h["TimeControl"]); cols["movetext"].append(mt); cols["plies"].append(plies)
            if len(cols["Site"]) >= BATCH: flush()
            if n % 5_000_000 == 0: log(f"  {os.path.basename(src)}: {n/1e6:.0f}M games read, {kept} kept")
    finally:
        flush(); writer.close()
    os.replace(tmp, dst)          # atomic: an interrupted run never leaves a half-written month
    return n, passed, kept, npos

# ---------------------------------------------------------------- download
def download(url, path, chunk=1 << 24):
    """Resumable download: continues a partial file with an HTTP Range request."""
    total = int(requests.head(url, timeout=60).headers["Content-Length"])
    have = os.path.getsize(path) if os.path.exists(path) else 0
    while have < total:
        with requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "ab") as f:
                try:
                    for part in r.iter_content(chunk):
                        f.write(part)
                except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
                    pass                              # connection dropped: loop and resume
        have = os.path.getsize(path)
    return path

def month_job(ym, out_dir, raw_dir):
    dst = os.path.join(out_dir, f"lichess_{ym}_filtered.parquet")
    if os.path.exists(dst):
        return ym, "skipped (already done)"
    raw = os.path.join(raw_dir, f"lichess_db_standard_rated_{ym}.pgn.zst")
    print(f"{ym}: downloading", flush=True)
    download(URL.format(ym), raw)
    print(f"{ym}: filtering", flush=True)
    n, passed, kept, npos = filter_file(raw, dst, log=lambda s: print(s, flush=True))
    os.remove(raw)
    return ym, f"{n:,} games read, {passed:,} pass headers, {kept:,} kept, {npos:,} qualifying positions"

def available_months(start):
    y, m = map(int, start.split("-"))
    months = []
    while True:
        ym = f"{y}-{m:02d}"
        if requests.head(URL.format(ym), timeout=60).status_code != 200:
            return months
        months.append(ym)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-08", help="first month, YYYY-MM")
    ap.add_argument("--out", default="chessdata")
    ap.add_argument("--workers", type=int, default=2, help="months processed in parallel (~31 GB disk each)")
    a = ap.parse_args()
    out_dir, raw_dir = os.path.join(a.out, "filtered"), os.path.join(a.out, "_raw")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(raw_dir, exist_ok=True)
    months = available_months(a.start)
    print(f"months available: {months[0]} .. {months[-1]} ({len(months)})", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(month_job, ym, out_dir, raw_dir): ym for ym in months}
        for fut in as_completed(futs):
            ym, msg = fut.result()
            print(f"DONE {ym}: {msg}", flush=True)

if __name__ == "__main__":      # required on Windows for multiprocessing
    main()
