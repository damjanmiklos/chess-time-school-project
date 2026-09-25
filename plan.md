# Clock-aware W/D/L prediction on Maia-3: project brief

School project (AI for Data Science). Fine-tune **Maia-3** to predict the **win/draw/loss outcome of human Lichess games** from a position, restricted to positions where one player has at least **2× the clock time** of the other. Most details are specified below; use your judgment for the rest.

## 1. Goal and model inputs

Predict P(win, draw, loss) from the side to move's perspective. The model gets exactly these inputs:

- The board with 8 positions of history (Maia-3's standard input)
- Both players' Elo (Maia-3's standard `self_elo` / `oppo_elo`)
- Both players' remaining clock time
- The time control's base time (games have different base times, so remaining clock alone is ambiguous)
- The Stockfish W/D/L evaluation of the position
- **** (Of course in the scratch folder you can write as complicated temporary code as you want, I don't care about that.)

Nothing else. All games have no increment, so increment isn't an input.

## 2. Environment

- **Machine:** Windows 11, Ryzen 9 5950X (16 cores/32 threads, Zen 3), RTX 3080 Ti (12 GB), 32 GB RAM.
- **Python environment:** a dedicated maia3 conda env with Python 3.11.
  - Check that `torch.cuda.is_available()` returns `True`.
- **Windows multiprocessing:** every entry point that uses multiprocessing or DataLoader workers needs an `if __name__ == "__main__":` guard.

## 3. Data

**Source:** the raw Lichess monthly dumps at `https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst`.

- Months: **August 2025 through the latest available** (August 2026 at time of writing).
- Each dump is about 30 GB.
- Don't use the Hugging Face mirror (`Lichess/standard-chess-games`): it stops at September 2025 and is about twice the size.

**Download script:** use the provided `pull_lichess_raw.py` as is. It downloads each month (resumable), stream-parses it, writes one filtered parquet file per month and deletes the raw dump. Its filters already match the list below.

**Filters.** Header-level:
- Rated games only.
- No player titled `BOT`.
- `Termination != "Abandoned"`. Keep time-forfeit games: they carry the clock signal.
- Base time ≥ 600 s with no increment (`TimeControl` of the form `N+0` with `N >= 600`). About 93% of these games are 600+0; 1800+0 and 900+0 are the next most common.
- `|WhiteElo - BlackElo| <= 150`.
- `%clk` annotations present.

Position-level: keep positions where `max(clock_white, clock_black) >= 2 * min(...)`.

**Clock semantics:**
- `%clk` is the remaining time of the player who just moved, measured after the move, in whole seconds.
- To get both clocks at the position before ply `i`, carry the opponent's last value forward.
- Both clocks start at the game's base time.
- An even ply index means White is to move.
- The script already does this and stores a `plies` list (indices of qualifying positions) for each game.

**Expected volume:**
- About 2M games per month pass all filters, so about 25M games in total.
- There are about 27 qualifying positions per game on average.

**Sampling and shuffling:**
- Sample **at most 5 random qualifying positions per game**. All positions from one game share the same outcome label, and too many per game leads to memorizing games.
- **Globally shuffle** with a two-pass bucket shuffle:
  1. Stream the samples into about 1,000 buckets chosen at random.
  2. Shuffle each bucket in memory and write it out as a training shard.
- A DataLoader shuffle buffer alone is not enough.

**Splits (by month, and therefore by game):**
- Train: August 2025 to June 2026.
- Validation: July 2026.
- Test: August 2026.
- Maia-3 was trained on January 2023 to July 2025, so there's no overlap.

**Label:** the game result converted to the side to move's perspective, as a 3-class target (W/D/L).

**Storage:** store compact rows (game moves, clocks, Elos, sampled ply index, Stockfish W/D/L) and build the Maia-3 tensors in the DataLoader workers. Pre-encoding every position as a tensor is far too large.

## 4. Stockfish labels

**Engine:** Stockfish 19, release tag `sf_19`. Do not use the `stockfish-dev-*` pre-releases. For Windows on x86-64 there is a single universal binary that picks the best instruction set for the CPU at runtime. The download needs no login or installer:

```
https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip
→ stockfish/stockfish-windows-x86-64-universal.exe
```

For labeling on the Linux HPC, use `stockfish-linux-x86-64-universal.tar.gz` from the same release.

**Verify the binary.** Run `<exe> bench` once and check that `Nodes searched` is **2497913**. That value was measured on the Linux build; builds of the same version should match. If it doesn't, stop and report instead of continuing. Pin this binary and use the same settings at training and inference time.

**Per position:**
```
setoption name Threads value 1
setoption name Hash value 16
setoption name UCI_ShowWDL value true
ucinewgame
position fen <fen>
go nodes 10000
```
- Wait for `bestmove`, then read the WDL from the last `info` line that contains `wdl` and is not marked `lowerbound`/`upperbound`. `python-chess`'s `chess.engine` with `Limit(nodes=10000)` handles this.
- Store the WDL in per mille, from the side to move's perspective. Don't use centipawns: WDL is bounded and handles mate scores without special cases.

**Parallelism:** run about 24–28 single-threaded engine processes, leaving some cores free.

**Deduplication:** key positions by pieces, side to move, castling rights and en passant square, and evaluate each unique position once.

**Cost:** 10k nodes takes about 10 ms per position per thread. About 125M positions is roughly half a day to a day on the 5950X.

Don't use Lichess's `%eval` annotations: they're only in about 6% of games and come from mixed engine settings. You need to always write the stockfish eval results to the file so that new runs don't need to rerun the stockfish evals, so replace the old %eval annotations with the new one (if it even has an old one). Also tag the positions with a flag that says if it has a stockfish evaluation so it's much faster and easire to know whether it's the old eval or perhaps no eval at all.

## 5. Model changes (Maia-3)

**Relevant facts about `maia3/models.py`:**
- `MAIA3Model.forward(tokens, self_elos, oppo_elos)` returns `(logits_move[B,4352], logits_value[B,3], logits_ponder[B])`.
- The **value head is already a human-outcome W/D/L head**, from the side to move's perspective.
- Elo enters only at the input: interpolated embeddings are concatenated onto every square token before `token_projection`.
- The transformer blocks are **post-norm** (`x = norm(x + sublayer(x))`), not pre-norm.
- Build the input tokens with `maia3/dataset.py` (`get_historical_tokens`) using the released config: `history=8`, `use_padding=True`, `include_time_info=False`. The last token column is a label column that `forward` slices off.

**The change:** a 7-dim context vector passes through a small MLP. The resulting embedding is injected in two places, each via **zero-initialized new weights**, so that the model behaves exactly like pretrained Maia-3 at step 0:
- concatenated onto every square token, inserted after the board features and before the Elo embeddings;
- concatenated onto the pooled features right before `fc_value_hid`.

Don't use DiT-style adaLN-Zero: its zero gates would switch off the pretrained post-norm blocks.

```python
def context_features(base, clk_self, clk_oppo, sf_wdl):    # seconds (B,), per-mille (B,3)
    ls, lo, lb = torch.log1p(clk_self), torch.log1p(clk_oppo), torch.log1p(base)
    return torch.cat([torch.stack([ls/7, lo/7, ls-lo, lb/7], -1), sf_wdl/1000.0], -1)   # (B,7)

def _widen(old, insert_at, n_new):                         # zero-init new input columns
    new = nn.Linear(old.in_features + n_new, old.out_features, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :insert_at] = old.weight[:, :insert_at]
        new.weight[:, insert_at+n_new:] = old.weight[:, insert_at:]
        if old.bias is not None: new.bias.copy_(old.bias)
    return new

class ClockMaia3(MAIA3Model):
    def add_context_inputs(self, d_ctx=32):                # call AFTER loading pretrained weights
        self.ctx_mlp = nn.Sequential(nn.Linear(7, d_ctx), nn.GELU(), nn.Linear(d_ctx, d_ctx))
        self.token_projection = _widen(self.token_projection, 12*self.cfg.history, d_ctx)
        self.fc_value_hid     = _widen(self.fc_value_hid, self.cfg.dim_vit, d_ctx)

    def forward(self, tokens, self_elos, oppo_elos, ctx):
        tokens = tokens[:, :, :12*self.cfg.history]
        c  = self.ctx_mlp(ctx)
        se = self.interpolate_elo(self_elos).unsqueeze(1).expand(-1, 64, -1)
        oe = self.interpolate_elo(oppo_elos).unsqueeze(1).expand(-1, 64, -1)
        x  = self.transformer(self.token_projection(
                torch.cat([tokens, c.unsqueeze(1).expand(-1, 64, -1), se, oe], -1)))
        logits_move = self._policy(x)      # move the original policy-head code into this helper
        pooled = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(torch.cat([pooled, c], -1))))
        logits_ponder = self.fc_ponder(F.relu(self.fc_ponder_hid(pooled))).squeeze(1)
        return logits_move, logits_value, logits_ponder
```

**Required test:** after `add_context_inputs()`, the move and value outputs must match the unmodified pretrained model to floating-point precision on a real batch.

## 6. Training

- **Model sizes:** prototype with **Maia3-5M** (hours per run on the 3080 Ti). The final model is **Maia3-23M**. Skip 79M: it's about 3× slower than 23M and only about 0.5 points more accurate.
- **Throughput (rough, on the 3080 Ti):**
  - 5M: about 1B positions/day.
  - 23M: about 250M positions/day.
  - Measure the real numbers in the first minutes of a run.
- **Epochs:** a single pass over the data is enough. Sampling covers more games; there's no need to repeat epochs.
- **Loss:** cross-entropy on the value head (W/D/L), plus **0.1–0.3 ×** cross-entropy of the policy head on the move actually played. The policy term keeps the shared layers from drifting away from their pretraining. Reuse Maia-3's move-index mapping (4352 = 64×64 from-to pairs + 256 promotions).
- **Schedule:**
  - Warm-up, about 5% of steps: freeze all pretrained parameters and train only `ctx_mlp`, the new weight columns and the value head.
  - Main phase: unfreeze everything. AdamW with two parameter groups; the pretrained weights get about **10× lower** learning rate than the new ones. Cosine or warmup-stable-decay schedule.
- **Precision:** bf16 autocast (the 3080 Ti supports it).
- **Batch size:** 1024–4096.
- **Checkpoints:** save regularly and log metrics.
- Use modern training techniques such as adamW etc.

## 7. Evaluation

Evaluate on the test months (July and August 2026). Report:
- log loss;
- Brier score;
- calibration (reliability diagram and ECE);
- accuracy.

Break the results down by clock-ratio bucket, Elo bucket, Stockfish-eval bucket and time control (600+0 vs. longer).

**Compare against:**
1. Logistic regression on Elo difference, log clock ratio, log base time and Stockfish expected score (W + D/2).
2. The **released Maia-3 value head** used as-is, with no clocks and no Stockfish input.
3. The full model.

The gap between 2 and 3 is the main result: how much the clocks and the Stockfish eval add.

**Ablations** (use the 5M model to keep them cheap):
- Full model **without the Stockfish input**. This shows whether the model just copies Stockfish.
- Where the context enters: input + head (default) vs. head only vs. conditioning every layer. For the last option, apply FiLM to each block's norm outputs, `norm(x+f(x))*(1+s)+b` with `s` and `b` zero-initialized.

Note for the report: because Stockfish is an input, the model is no longer strictly searchless.

## 8. Pitfalls checklist

- Using the CPU-only PyTorch wheel on Windows. Working in OneDrive.
- Using a different Stockfish version or node count at inference than in training.
- Shuffling only inside the DataLoader; splitting by position instead of by game and month.
- Too many positions per game, which leads to memorizing game outcomes.
- Getting the side-to-move perspective wrong for the label, the Stockfish W/D/L, or which Elo/clock is "self" vs "opponent".
- Breaking the exact-equivalence test at initialization. If it fails, a column was inserted in the wrong place.