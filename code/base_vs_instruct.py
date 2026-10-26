"""Matched base-vs-instruct circuit comparison for this repository's natural-lie setup.

This is deliberately a *two-model* extension, not a relabeling of the existing
"base" (unedited) column.  It learns a direction and a sparse head set separately
for each checkpoint, evaluates both on their own held-out items, uses matched
random head-set and random-direction controls, and—only when the architecture and
tokenizer are compatible—tests cross-checkpoint transfer.

Example (the matched official Qwen pair):
  python code/base_vs_instruct.py \
    --base-model Qwen/Qwen3.5-9B-Base --instruct-model Qwen/Qwen3.5-9B \
    --items data/items_roleplay_insider.json --out out/base_vs_instruct

The supplied data include Qwen-tokenizer answer IDs and reply prefixes.  For that
reason this script refuses to compare incompatible tokenizers rather than silently
scoring different tokens.  The two model IDs must also be architecture-compatible
for cross-transfer; native analyses still run if they are not.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import steer as S  # noqa: E402


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_items(path: str) -> list:
    value = json.load(open(os.path.expanduser(path), encoding="utf-8"))
    return value["items"] if isinstance(value, dict) else value


def tokenizer_signature(tok, items: list) -> dict:
    """Fingerprint the answer/prefix IDs which this repository's data reuses."""
    ids = sorted({x for it in items for key in ("lie_ids", "truth_ids", "prefix_ids")
                  for x in (it.get(key) or [])})
    return {"vocab_size": len(tok), "tokens": [tok.convert_ids_to_tokens(i) for i in ids]}


def compatible_direction(dst_eng, delta: dict) -> bool:
    return len(delta["h"]) == dst_eng.L and all(
        tuple(delta["h"][l].shape) == (dst_eng.nheads[l] * dst_eng.hdim[l],) for l in range(dst_eng.L)
    )


def make_random_heads(eng, heads: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    """Layer-matched null: replace each selected head with another head in its layer."""
    rng = np.random.default_rng(seed)
    selected = set(heads)
    out = []
    for layer, original in heads:
        free = [i for i in range(eng.nheads[layer]) if (layer, i) not in selected and (layer, i) not in out]
        out.append((layer, int(rng.choice(free))) if free else (layer, original))
    return out


def usable_recs(eng, tok, items, family, split, label, n_items, span):
    return S.nat_recs(eng, tok, items, family, [split], label, n_items, span)


def rank_and_select(eng, tok, items, a) -> tuple[dict, dict]:
    """Learn a natural direction, IG-rank heads, and empirically choose a short prefix."""
    # Direction construction needs the captured per-head span means; test scoring does not.
    select = {lab: S.nat_recs(eng, tok, items, a.family, ["select"], lab, a.n_select, a.span, mean=True)
              for lab in ("lie", "resist")}
    if not select["lie"] or not select["resist"]:
        raise RuntimeError("Need at least one usable select lie and select resist for each checkpoint.")
    delta, bins = S.delta_nat(select["lie"], select["resist"], a.bins)
    scaled = {k: [x * a.alpha for x in v] for k, v in delta.items()}
    widths = [eng.nheads[l] for l in range(eng.L)]
    all_heads = [(l, i) for l in range(eng.L) for i in range(widths[l])]
    scores = {}
    for label, sign in (("lie", 1.0), ("resist", -1.0)):
        acc, kept, changes = torch.zeros(eng.L, max(widths)), 0, []
        for rec in select[label]:
            run = type("Run", (), {"ids": rec["ids"], "lie": rec["lie"], "truth": rec["truth"]})()
            g = S.ig_units(eng, run, scaled, sign, a.ig_steps, "h")
            total = float(g.sum())
            changes.append(total)
            if abs(total) >= a.min_change:
                acc += g / total
                kept += 1
        if not kept:
            raise RuntimeError(f"No {label} selection item had an all-head change >= {a.min_change} bits.")
        scores[label] = acc / kept
        print(f"{label}: IG normalized over {kept}/{len(changes)} selection items; "
              f"median all-head change {np.median(changes):+.3f} bits", flush=True)
    indices_l = torch.tensor([x[0] for x in all_heads])
    indices_h = torch.tensor([x[1] for x in all_heads])
    combined = torch.minimum(scores["lie"], scores["resist"])
    order = [all_heads[j] for j in torch.argsort(-combined[indices_l, indices_h], stable=True).tolist()]

    sizes = E.grid(len(all_heads), a.per_doubling)
    masks = [S.masks(eng, [("h", l, h) for l, h in order[:k]]) for k in sizes]
    every = S.masks(eng, [("h", l, h) for l, h in all_heads])
    curves = {"sizes": sizes, "lie": [], "resist": []}
    for label, sign in (("lie", 1.0), ("resist", -1.0)):
        recs = select[label][:a.curve_items]
        rows = [S.nat_rel(eng, r, delta, masks + [every], a.alpha, a.span, sign)[:-1] for r in recs]
        curves[label] = np.nanmedian(np.array(rows), axis=0).round(5).tolist()
    lie_curve, resist_curve = np.array(curves["lie"]), np.array(curves["resist"])
    okay = (np.minimum(lie_curve, resist_curve) >= a.target_share) & \
           (np.maximum(lie_curve, resist_curve) <= a.max_share)
    prefix = sizes[int(np.argmax(okay))] if okay.any() else sizes[int(np.argmax(np.minimum(lie_curve, resist_curve)))]
    chosen = order[:prefix]
    summary = {
        "selection_counts": {k: len(v) for k, v in select.items()}, "matching_bins": bins,
        "n_heads": len(all_heads), "n_selected": prefix, "target_share": a.target_share,
        "max_share": a.max_share, "met_target": bool(okay.any()), "curves": curves,
        "heads": [{"layer": l, "index": h} for l, h in chosen],
        "ig": {k: scores[k].tolist() for k in scores},
    }
    return {"delta": delta, "heads": chosen}, summary


def score_arms(eng, tok, items, direction, heads, a, family=None) -> dict:
    """Held-out dose response with layer-matched head and norm-matched direction nulls."""
    family = family or a.family
    all_heads = [(l, i) for l in range(eng.L) for i in range(eng.nheads[l])]
    masks = {
        "selected": S.masks(eng, [("h", l, h) for l, h in heads]),
        "all_heads": S.masks(eng, [("h", l, h) for l, h in all_heads]),
        "random_heads": S.masks(eng, [("h", l, h) for l, h in make_random_heads(eng, heads, a.seed)]),
    }
    random_dir = S.random_delta(eng, direction, a.seed + 1)
    result = {"arms": list(masks), "alphas": a.alphas, "labels": {}}
    for label, sign in (("lie", 1.0), ("resist", -1.0)):
        recs = usable_recs(eng, tok, items, family, "test", label, a.n_test, a.span)
        label_out = {"n": len(recs), "arms": {}}
        if not recs:
            result["labels"][label] = label_out
            continue
        for arm, mask in masks.items():
            rows = [(sign * alpha, mask) for alpha in a.alphas]
            margins = []
            for rec in recs:
                lp = S.run_rows(eng, rec["ids"], S.Shift(eng, direction, rows, a.span, True))
                margins.append([eng.metric(row, rec["lie"], rec["truth"]) for row in lp])
            x = np.array(margins)
            label_out["arms"][arm] = {"share": ((x < 0) if label == "lie" else (x > 0)).mean(0).round(5).tolist(),
                                      "median_margin_bits": np.median(x, 0).round(5).tolist()}
        rows = [(sign * alpha, masks["selected"]) for alpha in a.alphas]
        margins = []
        for rec in recs:
            lp = S.run_rows(eng, rec["ids"], S.Shift(eng, random_dir, rows, a.span, True))
            margins.append([eng.metric(row, rec["lie"], rec["truth"]) for row in lp])
        x = np.array(margins)
        label_out["arms"]["random_direction"] = {
            "share": ((x < 0) if label == "lie" else (x > 0)).mean(0).round(5).tolist(),
            "median_margin_bits": np.median(x, 0).round(5).tolist(),
        }
        result["labels"][label] = label_out
    return result


def analyze_checkpoint(name: str, model_id: str, items: list, a, out: Path) -> dict:
    print(f"\n=== Loading {name}: {model_id} ===", flush=True)
    model, tok = E.load(model_id, a.dtype, a.device)
    eng = E.Engine(model)
    sig = tokenizer_signature(tok, items)
    learned, selection = rank_and_select(eng, tok, items, a)
    test = score_arms(eng, tok, items, learned["delta"], learned["heads"], a)
    torch.save({"delta": {k: [x.cpu() for x in v] for k, v in learned["delta"].items()}, "heads": learned["heads"]},
               out / f"{name}_circuit.pt")
    summary = {"model": model_id, "tokenizer": sig, "architecture": {"layers": eng.L, "heads": eng.nheads,
               "head_dims": eng.hdim}, "selection": selection, "native_test": test}
    dump_json(out / f"{name}.json", summary)
    del eng, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def cross_transfer(name: str, model_id: str, source: dict, items: list, a) -> dict:
    """Evaluate a source model's fixed locations/direction in a target model, if safely alignable."""
    state = torch.load(source["circuit"], weights_only=False)
    print(f"\n=== Cross-transfer {source['name']} -> {name} ===", flush=True)
    model, tok = E.load(model_id, a.dtype, a.device)
    eng = E.Engine(model)
    out = {"source": source["name"], "target": name, "ran": False}
    if tokenizer_signature(tok, items) != source["tokenizer"]:
        out["reason"] = "tokenizer fingerprint differs; stored answer IDs/prefixes are not portable"
    elif not compatible_direction(eng, state["delta"]) or any(h >= eng.nheads[l] for l, h in state["heads"]):
        out["reason"] = "head-output architecture differs; direct activation transfer is not dimensionally valid"
    else:
        out.update({"ran": True, "test": score_arms(eng, tok, items, state["delta"], state["heads"], a)})
    del eng, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def direction_similarity(base_path: Path, inst_path: Path) -> dict:
    a, b = torch.load(base_path, weights_only=False), torch.load(inst_path, weights_only=False)
    if len(a["delta"]["h"]) != len(b["delta"]["h"]) or any(x.shape != y.shape for x, y in zip(a["delta"]["h"], b["delta"]["h"])):
        return {"comparable": False, "reason": "head-output dimensions differ"}
    cos = [float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))
           for x, y in zip(a["delta"]["h"], b["delta"]["h"])]
    ah, bh = set(a["heads"]), set(b["heads"])
    return {"comparable": True, "per_layer_cosine": cos, "mean_layer_cosine": float(np.mean(cos)),
            "head_overlap": len(ah & bh), "head_jaccard": len(ah & bh) / len(ah | bh)}


def plot(summary: dict, out: Path) -> None:
    base, inst = summary["base"], summary["instruct"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"base": "#31688e", "instruct": "#b35c2e"}
    # Selection curves: circuit concentration in each model's own selection data.
    for name, rec in (("base", base), ("instruct", inst)):
        curve = rec["selection"]["curves"]
        axes[0, 0].plot(curve["sizes"], curve["lie"], color=colors[name], label=f"{name}: lies")
        axes[0, 0].plot(curve["sizes"], curve["resist"], "--", color=colors[name], label=f"{name}: resists")
    axes[0, 0].axhline(summary["config"]["target_share"], color="#333", ls=":", label="selection target")
    axes[0, 0].set_xscale("log"); axes[0, 0].set_ylim(0, 1.45); axes[0, 0].set_title("Native circuit-prefix curves")
    axes[0, 0].set_xlabel("top-ranked heads included"); axes[0, 0].set_ylabel("share of all-head effect"); axes[0, 0].legend(fontsize=8)
    # Dose response on held-out lies.
    for name, rec in (("base", base), ("instruct", inst)):
        data = rec["native_test"]["labels"].get("lie", {})
        for arm, style in (("selected", "-"), ("random_heads", "--"), ("random_direction", ":")):
            if data.get("arms", {}).get(arm):
                axes[0, 1].plot(rec["native_test"]["alphas"], data["arms"][arm]["share"], style,
                                color=colors[name], label=f"{name}: {arm}")
    axes[0, 1].set_title("Held-out lie removal dose response"); axes[0, 1].set_xlabel("steering coefficient α")
    axes[0, 1].set_ylabel("share with truth preferred"); axes[0, 1].set_ylim(0, 1.05); axes[0, 1].legend(fontsize=8)
    # Circuit size and native endpoint performance.
    names = ["base", "instruct"]; x = np.arange(2)
    axes[1, 0].bar(x - .18, [base["selection"]["n_selected"], inst["selection"]["n_selected"]], .36,
                   color=[colors[n] for n in names], label="selected heads")
    axes[1, 0].set_xticks(x, names); axes[1, 0].set_ylabel("head count"); axes[1, 0].set_title("Native selected-circuit size")
    ax = axes[1, 0].twinx()
    vals = []
    for r in (base, inst):
        lie = r["native_test"]["labels"].get("lie", {}).get("arms", {}).get("selected", {}).get("share", [])
        vals.append(lie[-1] if lie else np.nan)
    ax.plot(x, vals, "ko", label="held-out truth-preferred at max α"); ax.set_ylim(0, 1.05); ax.set_ylabel("held-out share")
    # Cross transfer and direction similarity summary.
    axes[1, 1].axis("off")
    sim = summary["direction_similarity"]
    lines = ["Cross-checkpoint summary", "", f"Direction comparable: {sim['comparable']}"]
    if sim["comparable"]:
        lines += [f"Mean per-layer cosine: {sim['mean_layer_cosine']:.3f}", f"Head Jaccard overlap: {sim['head_jaccard']:.3f}"]
    for key, value in summary["cross_transfer"].items():
        lines.append(f"{key}: {'ran' if value['ran'] else 'skipped'}")
        if not value["ran"]:
            lines.append(f"  {value['reason']}")
    axes[1, 1].text(.03, .95, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.suptitle("Matched base vs instruction/post-training circuit analysis", fontsize=15, fontweight="bold")
    fig.savefig(out / "base_vs_instruct.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="Raw/pretraining checkpoint identifier or local path")
    p.add_argument("--instruct-model", required=True, help="Matched post-trained checkpoint identifier or local path")
    p.add_argument("--items", default="data/items_roleplay_insider.json")
    p.add_argument("--out", default="out/base_vs_instruct")
    p.add_argument("--family", default="rp", help="Natural-lie family with select and test splits")
    p.add_argument("--dtype", default="float32"); p.add_argument("--device", default="cuda")
    p.add_argument("--span", type=int, default=9); p.add_argument("--bins", type=int, default=4)
    p.add_argument("--alpha", type=float, default=2.0); p.add_argument("--alphas", default="0.5,1,2")
    p.add_argument("--n-select", type=int, default=400); p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--ig-items", type=int, default=40); p.add_argument("--curve-items", type=int, default=40)
    p.add_argument("--ig-steps", type=int, default=6); p.add_argument("--per-doubling", type=int, default=4)
    p.add_argument("--target-share", type=float, default=.70); p.add_argument("--max-share", type=float, default=1.30)
    p.add_argument("--min-change", type=float, default=.05); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); a.alphas = [float(x) for x in a.alphas.split(",")]
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = load_items(a.items)
    base = analyze_checkpoint("base", a.base_model, items, a, out)
    inst = analyze_checkpoint("instruct", a.instruct_model, items, a, out)
    if base["tokenizer"] != inst["tokenizer"]:
        raise RuntimeError("Checkpoint tokenizers differ on the stored answer/prefix IDs. Refusing an invalid comparison.")
    base_state, inst_state = out / "base_circuit.pt", out / "instruct_circuit.pt"
    source_base = {"name": "base", "circuit": base_state, "tokenizer": base["tokenizer"]}
    source_inst = {"name": "instruct", "circuit": inst_state, "tokenizer": inst["tokenizer"]}
    cross = {"base_to_instruct": cross_transfer("instruct", a.instruct_model, source_base, items, a),
             "instruct_to_base": cross_transfer("base", a.base_model, source_inst, items, a)}
    result = {"config": {k: getattr(a, k) for k in ("base_model", "instruct_model", "family", "span", "bins", "alpha",
              "alphas", "target_share", "max_share", "ig_steps", "seed")}, "base": base, "instruct": inst,
              "direction_similarity": direction_similarity(base_state, inst_state), "cross_transfer": cross}
    dump_json(out / "base_vs_instruct.json", result)
    plot(result, out)
    print(f"Wrote {out / 'base_vs_instruct.json'} and {out / 'base_vs_instruct.png'}", flush=True)


if __name__ == "__main__":
    main()
