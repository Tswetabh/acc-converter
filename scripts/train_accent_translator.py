"""Train the Conformer Accent Translator on DTW-aligned HuBERT pairs."""
from __future__ import annotations

import argparse
import glob
import os
import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from accent_converter.models.backends.accent.conformer import ConformerAccentNet


def load_pairs(data_dir: str):
    items = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
        d = torch.load(f, weights_only=False)
        items.append((d["prompt_id"], d["src_orig"].float(), d["tgt_srclen"].float()))
    return items


def batches(items, bs, shuffle):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [items[j] for j in idx[i:i + bs]]
        lens = torch.tensor([x[1].shape[0] for x in b])
        src = pad_sequence([x[1] for x in b], batch_first=True)
        tgt = pad_sequence([x[2] for x in b], batch_first=True)
        yield src, tgt, lens


def loss_fn(pred, tgt, lens):
    mask = (torch.arange(pred.shape[1], device=pred.device)[None] < lens[:, None]).unsqueeze(-1)
    l1 = (F.l1_loss(pred, tgt, reduction="none") * mask).sum() / (mask.sum() * pred.shape[-1])
    cos = (1 - F.cosine_similarity(pred, tgt, dim=-1)) * mask.squeeze(-1)
    return l1 + cos.sum() / mask.sum(), l1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/aligned/arctic_train")
    p.add_argument("--output-dir", default="checkpoints/accent_translator")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    items = load_pairs(a.data_dir)
    if a.limit:
        items = items[:a.limit]
    prompts = sorted({x[0] for x in items})
    random.shuffle(prompts)
    val_prompts = set(prompts[: max(1, len(prompts) // 10)])  # split by sentence: no leakage
    train = [x for x in items if x[0] not in val_prompts]
    val = [x for x in items if x[0] in val_prompts]
    print(f"train={len(train)} val={len(val)}")

    kwargs = dict(dim=768, hidden=256, heads=4, ffn_dim=1024, layers=6, kernel=31, dropout=0.1)
    model = ConformerAccentNet(**kwargs).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    os.makedirs(a.output_dir, exist_ok=True)

    # baseline: passthrough (identity) val L1
    with torch.no_grad():
        tot, n = 0.0, 0
        for s, t, l in batches(val, a.batch_size, False):
            _, l1 = loss_fn(s.to(a.device), t.to(a.device), l.to(a.device))
            tot += l1.item(); n += 1
        print(f"passthrough baseline val L1: {tot / n:.4f}")

    best = float("inf")
    for ep in range(1, a.epochs + 1):
        model.train()
        for s, t, l in batches(train, a.batch_size, True):
            s, t, l = s.to(a.device), t.to(a.device), l.to(a.device)
            loss, _ = loss_fn(model(s, l), t, l)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for s, t, l in batches(val, a.batch_size, False):
                s, t, l = s.to(a.device), t.to(a.device), l.to(a.device)
                _, l1 = loss_fn(model(s, l), t, l)
                tot += l1.item(); n += 1
        v = tot / n
        print(f"epoch {ep}/{a.epochs} train_loss={loss.item():.4f} val_L1={v:.4f}", flush=True)
        ck = {"model": model.state_dict(), "model_kwargs": kwargs, "epoch": ep, "val_l1": v}
        torch.save(ck, os.path.join(a.output_dir, "latest.pt"))
        if v < best:
            best = v
            torch.save(ck, os.path.join(a.output_dir, "best.pt"))
    print(f"Done. best val L1 {best:.4f}")


if __name__ == "__main__":
    main()
