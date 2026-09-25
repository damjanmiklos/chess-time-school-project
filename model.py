"""Clock-aware Maia-3. New weights start at zero, so step 0 matches pretrained Maia-3."""
import math
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from maia3.models import MAIA3Model
from maia3.model_registry import resolve_model_spec, resolve_checkpoint_path


def context_features(base, clk_self, clk_oppo, sf_wdl):    # seconds (B,), per-mille (B,3)
    ls, lo, lb = torch.log1p(clk_self), torch.log1p(clk_oppo), torch.log1p(base)
    return torch.cat([torch.stack([ls / 7, lo / 7, ls - lo, lb / 7], -1), sf_wdl / 1000.0], -1)   # (B,7)


class WideLinear(nn.Module):
    """Same insertion as the brief's `_widen`, but the new columns are their own parameter.

    That lets the optimizer give the new columns a higher learning rate than the pretrained ones.
    """
    def __init__(self, old, insert_at, n_new):
        super().__init__()
        self.left = nn.Parameter(old.weight[:, :insert_at].detach().clone())
        self.fresh = nn.Parameter(old.weight.new_zeros(old.out_features, n_new))
        self.right = nn.Parameter(old.weight[:, insert_at:].detach().clone())
        self.bias = nn.Parameter(old.bias.detach().clone()) if old.bias is not None else None

    def forward(self, x):
        a = self.left.shape[1]
        b = a + self.fresh.shape[1]
        y = F.linear(x[..., :a], self.left) + F.linear(x[..., a:b], self.fresh) + F.linear(x[..., b:], self.right)
        return y if self.bias is None else y + self.bias


def _widen(old, insert_at, n_new):
    return WideLinear(old, insert_at, n_new)


class ClockMaia3(MAIA3Model):
    def add_context_inputs(self, d_ctx=32, where="both", use_sf=True):
        """where: 'both' (input + head), 'head', or 'film' (FiLM on every block + head)."""
        self.where = where
        self.use_sf = use_sf
        self.ctx_mlp = nn.Sequential(nn.Linear(7, d_ctx), nn.GELU(), nn.Linear(d_ctx, d_ctx))
        if where == "both":
            self.token_projection = _widen(self.token_projection, 12 * self.cfg.history, d_ctx)
        self.fc_value_hid = _widen(self.fc_value_hid, self.cfg.dim_vit, d_ctx)
        if where == "film":
            dim = self.cfg.dim_vit
            self.film = nn.ModuleList(nn.Linear(d_ctx, 4 * dim) for _ in self.transformer.layers)
            for layer in self.film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            for i, block in enumerate(self.transformer.layers):
                block.forward = self._film_forward(block, i)

    def _film_forward(self, block, index):
        dim = self.cfg.dim_vit

        def forward(x, attn_mask=None):
            s1, b1, s2, b2 = self._film_now[index].split(dim, dim=-1)
            sa_out, _ = block.self_attn(query=x, key=x, value=x, attn_mask=attn_mask)
            h = block.norm1(x + block.dropout1(sa_out))
            h = h * (1 + s1[:, None, :]) + b1[:, None, :]
            ff = block.linear2(block.dropout(block.activation(block.linear1(h))))
            h = block.norm2(h + block.dropout2(ff))
            return h * (1 + s2[:, None, :]) + b2[:, None, :]
        return forward

    def _policy(self, x):
        sq_from = self.proj_sq_from(x[:, :64, :])
        sq_to = self.proj_sq_to(x[:, :64, :])
        scores = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(self.cfg.head_hid_dim)
        flat = scores.reshape(x.size(0), 64 * 64)
        rank7 = [chess.square(f, 6) for f in range(8)]
        rank8 = [chess.square(f, 7) for f in range(8)]
        promo = self.promo_bias_proj(sq_to[:, rank8, :]) * math.sqrt(self.cfg.head_hid_dim)
        extra = []
        for from_file in range(8):
            for to_file in range(8):
                base = scores[:, rank7[from_file], rank8[to_file]]
                for piece in range(4):
                    extra.append((base + promo[:, to_file, piece]).unsqueeze(1))
        return torch.cat([flat, torch.cat(extra, dim=1)], dim=1)

    def forward(self, tokens, self_elos, oppo_elos, ctx):
        tokens = tokens[:, :, :12 * self.cfg.history]
        if not self.use_sf:
            ctx = torch.cat([ctx[:, :4], torch.zeros_like(ctx[:, 4:])], -1)
        c = self.ctx_mlp(ctx)
        se = self.interpolate_elo(self_elos).unsqueeze(1).expand(-1, 64, -1)
        oe = self.interpolate_elo(oppo_elos).unsqueeze(1).expand(-1, 64, -1)
        board = tokens if self.where != "both" else torch.cat([tokens, c.unsqueeze(1).expand(-1, 64, -1)], -1)
        x = self.token_projection(torch.cat([board, se, oe], -1))
        if hasattr(self, "abs_pe"):
            x = self.abs_pe(x)
        if self.where == "film":
            self._film_now = [layer(c) for layer in self.film]
        x = self.transformer(x)
        logits_move = self._policy(x)
        pooled = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(torch.cat([pooled, c], -1))))
        logits_ponder = self.fc_ponder(F.relu(self.fc_ponder_hid(pooled))).squeeze(1)
        return logits_move, logits_value, logits_ponder


def split_params(model):
    """New columns, the context MLP, FiLM, and the value head vs. everything pretrained."""
    high, low = [], []
    for name, param in model.named_parameters():
        new = "fresh" in name or name.startswith(("ctx_mlp", "film", "fc_value."))
        (high if new else low).append(param)
    return high, low


def maia_cfg(size):
    spec = resolve_model_spec(size)
    return SimpleNamespace(**spec.config), spec


def load_pretrained(model, spec):
    path = resolve_checkpoint_path(spec)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {(k[7:] if k.startswith("module.") else k).replace("smolgen", "gab"): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"pretrained load mismatch, missing={missing[:8]} unexpected={unexpected[:8]}")
    return path


def build(size="5m", where="both", use_sf=True, checkpoint=None):
    cfg, spec = maia_cfg(size)
    model = ClockMaia3(cfg)
    if checkpoint:
        model.add_context_inputs(where=where, use_sf=use_sf)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        load_pretrained(model, spec)
        model.add_context_inputs(where=where, use_sf=use_sf)
    return model


def _real_batch(cfg, n=4):
    """A few real clock-imbalance positions, tokenized the way Maia-3 expects."""
    import pyarrow.parquet as pq
    from config import RAW
    from data import encode_game
    path = next(RAW.glob("lichess_*_filtered.parquet"))
    table = pq.ParquetFile(path).read_row_group(0).slice(0, 20)
    rows = []
    for game in table.to_pylist():
        rows.extend(encode_game(game, list(range(len(game["plies"]))), cfg, limit=1))
        if len(rows) >= n:
            break
    batch = rows[:n]
    return {
        "tokens": torch.stack([r["tokens"] for r in batch]),
        "self_elo": torch.stack([r["self_elo"] for r in batch]),
        "oppo_elo": torch.stack([r["oppo_elo"] for r in batch]),
        "ctx": torch.randn(len(batch), 7),
    }


@torch.no_grad()
def check_equivalent(size="5m"):
    """Move and value logits must match unmodified Maia-3 after the new inputs are added."""
    cfg, spec = maia_cfg(size)
    ref = MAIA3Model(cfg).eval()
    load_pretrained(ref, spec)
    base = {k: v.clone() for k, v in ref.state_dict().items()}
    batch = _real_batch(cfg)
    ref_move, ref_value, _ = ref(batch["tokens"], batch["self_elo"], batch["oppo_elo"])
    for where in ("both", "head", "film"):
        model = ClockMaia3(cfg).eval()
        model.load_state_dict(base)
        model.add_context_inputs(where=where)
        move, value, _ = model(batch["tokens"], batch["self_elo"], batch["oppo_elo"], batch["ctx"])
        ok_m = torch.allclose(move, ref_move, rtol=1e-5, atol=1e-5)
        ok_v = torch.allclose(value, ref_value, rtol=1e-5, atol=1e-5)
        print(f"{where:5}  move max { (move - ref_move).abs().max().item():.3e}  "
              f"value max {(value - ref_value).abs().max().item():.3e}  match {ok_m and ok_v}")
        if not (ok_m and ok_v):
            raise SystemExit(f"equivalence failed for where={where}")
    print("equivalence ok")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    print("cuda", torch.cuda.get_device_name(0))
    check_equivalent("5m")
