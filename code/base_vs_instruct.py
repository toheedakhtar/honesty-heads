"""Matched base-vs-instruct circuit comparison for this repository's natural-lie setup.

This is deliberately a *two-model* extension, not a relabeling of the existing
"base" (unedited) column.  It learns a direction and a sparse head set separately
for each checkpoint, evaluates both on their own held-out items, uses matched
random head-set and random-direction controls, and—only when the architecture and
tokenizer are compatible—tests cross-checkpoint transfer.

Example (the matched official Qwen pair):
  python code/base_vs_instruct.py \
    --base-model Qwen/Qwen3.5-9B-Base --instruct-model Qwen/Qwen3.5-9B \
    --items data/items_roleplay_insider.json --out out/base_vs_instruct_v2 \
    --random 19

Then run the expensive behavioral/persona extensions without repeating the core:
  python code/base_vs_instruct.py \
    --base-model Qwen/Qwen3.5-9B-Base --instruct-model Qwen/Qwen3.5-9B \
    --out out/base_vs_instruct_v2 --extensions all --extensions-only

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
import subprocess
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
import answer_split as AS  # noqa: E402


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
    sizes = E.grid(len(all_heads), a.per_doubling)
    every = S.masks(eng, [("h", l, h) for l, h in all_heads])
    combined = {"min": torch.minimum(scores["lie"], scores["resist"]),
                "mean": (scores["lie"] + scores["resist"]) / 2}
    orders = {rule: [all_heads[j] for j in torch.argsort(-value[indices_l, indices_h], stable=True).tolist()]
              for rule, value in combined.items()}
    all_curves, candidates = {}, []
    for rule, order in orders.items():
        masks = [S.masks(eng, [("h", l, h) for l, h in order[:k]]) for k in sizes]
        curves = {"sizes": sizes, "lie": [], "resist": []}
        for label, sign in (("lie", 1.0), ("resist", -1.0)):
            recs = select[label][:a.curve_items]
            rows = [S.nat_rel(eng, r, delta, masks + [every], a.alpha, a.span, sign)[:-1] for r in recs]
            curves[label] = np.nanmedian(np.array(rows), axis=0).round(5).tolist()
        lie_curve, resist_curve = np.array(curves["lie"]), np.array(curves["resist"])
        joint = np.minimum(lie_curve, resist_curve)
        okay = (joint >= a.target_share) & (np.maximum(lie_curve, resist_curve) <= a.max_share)
        prefix = sizes[int(np.argmax(okay))] if okay.any() else sizes[int(np.nanargmax(joint))]
        candidates.append((not bool(okay.any()), prefix, -float(np.nanmax(joint)), rule))
        curves["first_passing"] = prefix if okay.any() else None
        curves["met_target"] = bool(okay.any())
        all_curves[rule] = curves
    _failed, prefix, _best, selected_rule = min(candidates)
    chosen = orders[selected_rule][:prefix]
    summary = {
        "selection_counts": {k: len(v) for k, v in select.items()}, "matching_bins": bins,
        "n_heads": len(all_heads), "n_selected": prefix, "target_share": a.target_share,
        "max_share": a.max_share, "met_target": not _failed, "rule": selected_rule, "curves": all_curves,
        "heads": [{"layer": l, "index": h} for l, h in chosen],
        "ig": {k: scores[k].tolist() for k in scores},
    }
    return {"delta": delta, "heads": chosen}, summary


def exact_rank_p(observed: float, null: list[float]) -> float | None:
    return None if not null else float((1 + sum(x >= observed for x in null)) / (len(null) + 1))


def summarize_matrix(recs: list, margins: np.ndarray, label: str, include_items: bool = True) -> dict:
    """Summarize [items, alphas] margins with exact rate and bootstrap median intervals."""
    baseline = np.array([r["m"] for r in recs])
    sign = 1.0 if label == "lie" else -1.0
    hit = margins < 0 if label == "lie" else margins > 0
    improvement = sign * (baseline[:, None] - margins)
    out = {
        "share": hit.mean(0).round(5).tolist(),
        "share_ci": [S.share_ci(hit[:, j].astype(float)) for j in range(margins.shape[1])],
        "median_margin_bits": np.median(margins, 0).round(5).tolist(),
        "median_improvement_bits": np.median(improvement, 0).round(5).tolist(),
        "median_improvement_ci": [[round(float(x), 5) for x in E.boot_median(improvement[:, j])]
                                  for j in range(margins.shape[1])],
    }
    if include_items:
        out["items"] = [{"id": r["it"]["id"], "baseline_margin": round(float(baseline[i]), 6),
                         "margins": margins[i].round(6).tolist(), "improvement": improvement[i].round(6).tolist(),
                         "hit": hit[i].astype(int).tolist(),
                         "polarity": AS.polarity(r["it"]["lie"] if label == "lie" else r["it"]["truth"])}
                        for i, r in enumerate(recs)]
    return out


def yes_bias(recs: list, matrices: dict[str, np.ndarray], nulls: dict[str, list[np.ndarray]], label: str) -> dict:
    """Truth-directed versus generic-Yes decomposition at every alpha."""
    if not recs:
        return {}
    baseline = np.array([r["m"] for r in recs])
    pol = np.array([AS.polarity(r["it"]["lie"] if label == "lie" else r["it"]["truth"]) for r in recs])
    sign = 1.0 if label == "lie" else -1.0

    def one(x):
        split = AS.split(baseline, x, pol, sign)
        return {k: np.asarray(split[k]).round(5).tolist() for k in ("honesty", "yes_push", "yes", "no", "other")}
    out = {name: one(x) for name, x in matrices.items()}
    out["nulls"] = {name: [one(x) for x in xs] for name, xs in nulls.items()}
    out["counts"] = {k: int((pol == k).sum()) for k in ("yes", "no", "other")}
    return out


def score_arms(eng, tok, items, direction, heads, a, family=None, n_random=None) -> dict:
    """Held-out dose response with repeated matched head-set and direction nulls."""
    family, n_random = family or a.family, a.random if n_random is None else n_random
    all_heads = [(l, i) for l in range(eng.L) for i in range(eng.nheads[l])]
    masks = {"selected": S.masks(eng, [("h", l, h) for l, h in heads]),
             "all_heads": S.masks(eng, [("h", l, h) for l, h in all_heads])}
    random_masks = [S.masks(eng, [("h", l, h) for l, h in make_random_heads(eng, heads, a.seed + q + 1)])
                    for q in range(n_random)]
    random_dirs = [S.random_delta(eng, direction, a.seed + 1000 + q) for q in range(n_random)]
    result = {"alphas": a.alphas, "n_random": n_random, "labels": {}}
    for label, sign in (("lie", 1.0), ("resist", -1.0)):
        recs = usable_recs(eng, tok, items, family, "test", label, a.n_test, a.span)
        label_out = {"n": len(recs), "arms": {}, "nulls": {"random_heads": [], "random_directions": []}}
        if not recs:
            result["labels"][label] = label_out
            continue

        def evaluate(dl, mask):
            rows = [(sign * alpha, mask) for alpha in a.alphas]
            values = []
            for rec in recs:
                lp = S.run_rows(eng, rec["ids"], S.Shift(eng, dl, rows, a.span, True))
                values.append([eng.metric(row, rec["lie"], rec["truth"]) for row in lp])
            return np.array(values)

        matrices = {name: evaluate(direction, mask) for name, mask in masks.items()}
        for name, x in matrices.items():
            label_out["arms"][name] = summarize_matrix(recs, x, label)
        null_matrices = {"random_heads": [evaluate(direction, mask) for mask in random_masks],
                         "random_directions": [evaluate(dl, masks["selected"]) for dl in random_dirs]}
        for kind, xs in null_matrices.items():
            label_out["nulls"][kind] = [summarize_matrix(recs, x, label, False) for x in xs]
        observed = label_out["arms"]["selected"]
        label_out["randomization_p"] = {}
        for kind, vals in label_out["nulls"].items():
            label_out["randomization_p"][kind] = {
                "share": [exact_rank_p(observed["share"][j], [x["share"][j] for x in vals]) for j in range(len(a.alphas))],
                "median_improvement": [exact_rank_p(observed["median_improvement_bits"][j],
                                                     [x["median_improvement_bits"][j] for x in vals])
                                       for j in range(len(a.alphas))],
            }
        label_out["yes_bias"] = yes_bias(recs, matrices, null_matrices, label)
        result["labels"][label] = label_out
    return result


def score_fixed_test(eng, tok, items, direction, heads, a) -> dict:
    """Score identical stored test IDs in both models, without conditioning on each model's baseline sign."""
    mask = S.masks(eng, [("h", l, h) for l, h in heads])
    out = {"alphas": a.alphas, "labels": {}}
    for label, sign in (("lie", 1.0), ("resist", -1.0)):
        candidates = [it for it in items if it.get("family") == a.family and it.get("split") == "test"
                      and it.get("label") == label and it.get("usable")]
        recs = []
        for it in candidates:
            rec = S.nat_rec(eng, tok, it, a.span, False)
            if rec is not None:
                recs.append(rec)
            if len(recs) >= a.n_test:
                break
        margins = []
        rows = [(sign * alpha, mask) for alpha in a.alphas]
        for rec in recs:
            lp = S.run_rows(eng, rec["ids"], S.Shift(eng, direction, rows, a.span, True))
            margins.append([eng.metric(row, rec["lie"], rec["truth"]) for row in lp])
        out["labels"][label] = {"n": len(recs), "selected": summarize_matrix(recs, np.array(margins), label)
                                if recs else {}}
    return out


def analyze_checkpoint(name: str, model_id: str, items: list, a, out: Path) -> dict:
    print(f"\n=== Loading {name}: {model_id} ===", flush=True)
    model, tok = E.load(model_id, a.dtype, a.device)
    eng = E.Engine(model)
    sig = tokenizer_signature(tok, items)
    learned, selection = rank_and_select(eng, tok, items, a)
    test = score_arms(eng, tok, items, learned["delta"], learned["heads"], a)
    fixed_test = score_fixed_test(eng, tok, items, learned["delta"], learned["heads"], a)
    eligibility = {split: {label: {"stored_candidates": sum(it.get("family") == a.family and
                            it.get("split") == split and it.get("label") == label and it.get("usable") for it in items),
                            "remeasured_with_expected_sign": (selection["selection_counts"][label] if split == "select"
                                                              else test["labels"][label]["n"])}
                           for label in ("lie", "resist")} for split in ("select", "test")}
    cpu_delta = {k: [x.cpu() for x in v] for k, v in learned["delta"].items()}
    circuit_path, delta_path, heads_path = (out / f"{name}_circuit.pt", out / f"{name}_delta.pt",
                                            out / f"{name}_heads.json")
    torch.save({"delta": cpu_delta, "heads": learned["heads"]}, circuit_path)
    torch.save(cpu_delta, delta_path)
    dump_json(heads_path, {"P": a.span, "source": [a.family], "delta": str(delta_path.resolve()),
                           "select_alpha": a.alpha, "rank": "integrated gradients",
                           "rule": selection["rule"], "size": len(learned["heads"]),
                           "heads": [{"layer": l, "index": h, "label": f"L{l}.H{h}"}
                                     for l, h in learned["heads"]]})
    summary = {"model": model_id, "tokenizer": sig, "architecture": {"layers": eng.L, "heads": eng.nheads,
               "head_dims": eng.hdim}, "artifacts": {"circuit": str(circuit_path.resolve()),
               "delta": str(delta_path.resolve()), "heads": str(heads_path.resolve())},
               "eligibility": eligibility, "selection": selection, "native_test": test, "fixed_test": fixed_test}
    dump_json(out / f"{name}.json", summary)
    del eng, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def direction_norm(delta: dict) -> float:
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in delta["h"])))


def scaled_direction(source: dict, target: dict, eng, mode: str, factor: float = 1.0) -> dict:
    """Put a source direction on a target-comparable activation scale."""
    if mode in ("raw", "behavior"):
        scale = factor
        return {k: [x * scale for x in source[k]] for k in source}
    if mode == "global_norm":
        scale = direction_norm(target) / max(direction_norm(source), 1e-12)
        return {k: [x * scale for x in source[k]] for k in source}
    if mode != "per_head_norm":
        raise ValueError(mode)
    out = {"h": [], "n": []}
    global_scale = direction_norm(target) / max(direction_norm(source), 1e-12)
    for l, (src, dst) in enumerate(zip(source["h"], target["h"])):
        s = src.view(eng.nheads[l], eng.hdim[l]).clone()
        d = dst.view(eng.nheads[l], eng.hdim[l])
        scale = d.float().norm(dim=1) / s.float().norm(dim=1).clamp_min(1e-12)
        out["h"].append((s * scale.to(s.dtype)[:, None]).flatten())
    out["n"] = [x * global_scale for x in source["n"]]
    return out


def median_selection_effect(eng, tok, items, delta, a) -> float:
    """Median all-head improvement on target selection lies at the reference alpha."""
    recs = usable_recs(eng, tok, items, a.family, "select", "lie", a.n_select, a.span)
    heads = [(l, i) for l in range(eng.L) for i in range(eng.nheads[l])]
    mask = S.masks(eng, [("h", l, h) for l, h in heads])
    changes = []
    for rec in recs[:a.curve_items]:
        row = S.run_rows(eng, rec["ids"], S.Shift(eng, delta, [(a.alpha, mask)], a.span, True))[0]
        changes.append(rec["m"] - eng.metric(row, rec["lie"], rec["truth"]))
    return float(np.median(changes)) if changes else float("nan")


def cross_transfer(name: str, model_id: str, source: dict, target_circuit: Path, items: list, a) -> dict:
    """Evaluate raw and calibrated source circuits in an aligned target checkpoint."""
    state = torch.load(source["circuit"], weights_only=False)
    target = torch.load(target_circuit, weights_only=False)
    print(f"\n=== Cross-transfer {source['name']} -> {name} ===", flush=True)
    model, tok = E.load(model_id, a.dtype, a.device)
    eng = E.Engine(model)
    out = {"source": source["name"], "target": name, "ran": False}
    if tokenizer_signature(tok, items) != source["tokenizer"]:
        out["reason"] = "tokenizer fingerprint differs; stored answer IDs/prefixes are not portable"
    elif not compatible_direction(eng, state["delta"]) or any(h >= eng.nheads[l] for l, h in state["heads"]):
        out["reason"] = "head-output architecture differs; direct activation transfer is not dimensionally valid"
    else:
        native_effect = median_selection_effect(eng, tok, items, target["delta"], a)
        source_effect = median_selection_effect(eng, tok, items, state["delta"], a)
        behavior_factor = abs(native_effect / source_effect) if np.isfinite(source_effect) and abs(source_effect) > 1e-9 else 1.0
        modes = {mode: scaled_direction(state["delta"], target["delta"], eng, mode, behavior_factor)
                 for mode in ("raw", "global_norm", "per_head_norm", "behavior")}
        out.update({"ran": True, "calibration": {"native_selection_effect": native_effect,
                    "raw_source_selection_effect": source_effect, "behavior_factor": behavior_factor,
                    "source_norm": direction_norm(state["delta"]), "target_norm": direction_norm(target["delta"])},
                    "modes": {mode: score_arms(eng, tok, items, dl, state["heads"], a,
                                                n_random=a.cross_random)
                              for mode, dl in modes.items()}})
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


def paired_permutation_p(difference: np.ndarray, seed: int, draws: int = 20000) -> float:
    """Two-sided paired sign-flip test of a mean difference."""
    if not len(difference):
        return float("nan")
    observed = abs(float(difference.mean()))
    rng = np.random.default_rng(seed)
    extreme, done = 0, 0
    while done < draws:
        n = min(2000, draws - done)
        signs = rng.choice((-1.0, 1.0), size=(n, len(difference)))
        extreme += int((np.abs((signs * difference).mean(1)) >= observed - 1e-12).sum())
        done += n
    return float((extreme + 1) / (draws + 1))


def shared_comparison(base: dict, instruct: dict, a) -> dict:
    """Paired comparison on identical test IDs, avoiding model-specific eligibility conditioning."""
    from scipy.stats import binomtest
    out = {"note": "fixed stored test IDs; no filtering by either model's baseline margin sign", "labels": {}}
    for label in ("lie", "resist"):
        bitems = {x["id"]: x for x in base["fixed_test"]["labels"][label]["selected"].get("items", [])}
        iitems = {x["id"]: x for x in instruct["fixed_test"]["labels"][label]["selected"].get("items", [])}
        ids = sorted(bitems.keys() & iitems.keys())
        rows = []
        for j, alpha in enumerate(a.alphas):
            bimp = np.array([bitems[k]["improvement"][j] for k in ids])
            iimp = np.array([iitems[k]["improvement"][j] for k in ids])
            diff = bimp - iimp
            rng = np.random.default_rng(a.seed + j)
            boot = np.median(diff[rng.integers(0, len(diff), (5000, len(diff)))], 1) if len(diff) else np.array([])
            bhit = np.array([bitems[k]["hit"][j] for k in ids], bool)
            ihit = np.array([iitems[k]["hit"][j] for k in ids], bool)
            b_only, i_only = int((bhit & ~ihit).sum()), int((ihit & ~bhit).sum())
            rows.append({"alpha": alpha, "n": len(ids), "base_mean_improvement": float(bimp.mean()) if len(ids) else None,
                         "instruct_mean_improvement": float(iimp.mean()) if len(ids) else None,
                         "mean_difference_base_minus_instruct": float(diff.mean()) if len(ids) else None,
                         "median_difference": float(np.median(diff)) if len(ids) else None,
                         "median_difference_bootstrap_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
                         if len(boot) else [None, None],
                         "paired_signflip_p": paired_permutation_p(diff, a.seed + j) if len(ids) else None,
                         "base_hit": int(bhit.sum()), "instruct_hit": int(ihit.sum()),
                         "mcnemar_discordant": [b_only, i_only],
                         "mcnemar_p": float(binomtest(b_only, b_only + i_only, .5).pvalue)
                         if b_only + i_only else 1.0})
        out["labels"][label] = {"ids": ids, "alphas": rows}
    return out


def run_command(cmd: list[str], cwd: Path) -> None:
    print("\n+ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run_extensions(model_name: str, model_id: str, summary: dict, a, root: Path, out: Path) -> dict:
    """Run expensive existing-repository evaluations as resumable subprocess stages."""
    requested = set(a.extensions)
    if "all" in requested:
        requested = {"persona", "audit", "ablation", "sycophancy"}
    if not requested:
        return {}
    py = sys.executable
    ext = out / "extensions" / model_name
    ext.mkdir(parents=True, exist_ok=True)
    heads, delta = summary["artifacts"]["heads"], summary["artifacts"]["delta"]
    result = {}
    controls = str((root / a.controls).resolve())
    items = str((root / a.items).resolve())
    ctl_groups = "honest_tf,honest_qa,french,caps,json,sys_caps,sys_negation,sys_fiction,sys_labelswap,sys_swap_shots,sys_honest"

    if "persona" in requested:
        dest = ext / "persona"; dest.mkdir(exist_ok=True)
        run_command([py, "code/steer.py", "natpersona", "--model", model_id, "--dtype", a.dtype,
                     "--items", a.persona_items, "--heads", heads, "--delta", delta,
                     "--random", str(a.persona_random), "--out", str(dest)], root)
        run_command([py, "code/persona_stats.py", str(dest / "persona.json"), str(dest / "persona_stats.json")], root)
        result["persona"] = str((dest / "persona_stats.json").resolve())

    if "ablation" in requested:
        dest = ext / "ablation"; dest.mkdir(exist_ok=True)
        run_command([py, "code/steer.py", "natablate", "--model", model_id, "--dtype", a.dtype,
                     "--items", items, "--heads", heads, "--delta", delta, "--families", a.family,
                     "--labels", "lie,resist", "--n-items", str(a.n_test), "--n-select", "64",
                     "--random", str(a.random), "--controls", controls, "--ctl-groups", ctl_groups,
                     "--out", str(dest)], root)
        result["ablation_and_controls"] = str((dest / "ablate_nat.json").resolve())

    if "audit" in requested:
        dest = ext / "audit"; dest.mkdir(exist_ok=True)
        core = dest / "core.json"
        run_command([py, "code/make_core.py", heads, delta, str(core)], root)
        base_file = dest / "base.json"
        if not base_file.exists():
            dump_json(base_file, {"lies": [], "controls": []})
        run_command([py, "code/audit_eval.py", "free", "--model", model_id, "--items", a.audit_items,
                     "--controls", controls, "--out", str(dest), "--dtype", a.dtype, "--core", str(core),
                     "--free-items", a.audit_items, "--families", "it", "--tag", "it", "--rand-free", "2",
                     "--rand-modes", "every", "--free-max-new", "512", "--free-lies-only"], root)
        result["insider_audit"] = str(dest.resolve())

    if "sycophancy" in requested:
        dest = ext / "sycophancy"; natural = dest / "natural"; selected = dest / "selected"
        natural.mkdir(parents=True, exist_ok=True); selected.mkdir(parents=True, exist_ok=True)
        run_command([py, "code/steer.py", "natural", "--model", model_id, "--dtype", a.dtype,
                     "--items", items, "--source", "syco_v2", "--families", "syco_v2,rp", "--out", str(natural),
                     "--span", str(a.span), "--alphas", "1,2", "--wheres", "every", "--random-deltas", str(a.random),
                     "--bins", str(a.bins), "--n-select", str(a.n_select), "--n-items", str(a.n_test),
                     "--controls", controls, "--ctl-groups", ctl_groups], root)
        syco_delta = natural / "delta_A_syco_v2.pt"
        run_command([py, "code/steer.py", "natsel", "--model", model_id, "--dtype", a.dtype,
                     "--items", items, "--delta", str(syco_delta), "--source", "syco_v2",
                     "--families", "syco_v2,rp", "--out", str(selected), "--span", str(a.span),
                     "--select-alpha", str(a.alpha), "--ig-items", str(a.ig_items), "--ig-steps", str(a.ig_steps),
                     "--curve-items", str(a.curve_items), "--per-doubling", str(a.per_doubling),
                     "--random", str(a.random), "--n-items", str(a.n_test), "--controls", controls,
                     "--ctl-groups", ctl_groups], root)
        result["sycophancy"] = str(selected.resolve())
    return result


def plot(summary: dict, out: Path) -> None:
    base, inst = summary["base"], summary["instruct"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"base": "#31688e", "instruct": "#b35c2e"}
    # Selection curves: circuit concentration in each model's own selection data.
    for name, rec in (("base", base), ("instruct", inst)):
        curve = rec["selection"]["curves"][rec["selection"]["rule"]]
        axes[0, 0].plot(curve["sizes"], curve["lie"], color=colors[name], label=f"{name}: lies")
        axes[0, 0].plot(curve["sizes"], curve["resist"], "--", color=colors[name], label=f"{name}: resists")
    axes[0, 0].axhline(summary["config"]["target_share"], color="#333", ls=":", label="selection target")
    axes[0, 0].set_xscale("log"); axes[0, 0].set_ylim(0, 1.45); axes[0, 0].set_title("Native circuit-prefix curves")
    axes[0, 0].set_xlabel("top-ranked heads included"); axes[0, 0].set_ylabel("share of all-head effect"); axes[0, 0].legend(fontsize=8)
    # Dose response on held-out lies.
    for name, rec in (("base", base), ("instruct", inst)):
        data = rec["native_test"]["labels"].get("lie", {})
        for arm, style in (("selected", "-"), ("all_heads", "--")):
            if data.get("arms", {}).get(arm):
                axes[0, 1].plot(rec["native_test"]["alphas"], data["arms"][arm]["share"], style,
                                color=colors[name], label=f"{name}: {arm}")
        for kind, style in (("random_heads", ":"), ("random_directions", "-.")):
            nulls = data.get("nulls", {}).get(kind, [])
            if nulls:
                y = np.median(np.array([x["share"] for x in nulls]), 0)
                axes[0, 1].plot(rec["native_test"]["alphas"], y, style, color=colors[name], alpha=.65,
                                label=f"{name}: median {kind}")
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

    # Diagnostics: null distributions, polarity decomposition, paired comparison, calibrated transfer.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for pos, (name, rec) in enumerate((("base", base), ("instruct", inst))):
        data = rec["native_test"]["labels"]["lie"]
        obs = data["arms"]["selected"]["share"][-1]
        hr = [x["share"][-1] for x in data["nulls"]["random_heads"]]
        dr = [x["share"][-1] for x in data["nulls"]["random_directions"]]
        axes[0, 0].scatter([pos - .12] * len(hr), hr, color=colors[name], alpha=.55, marker="o")
        axes[0, 0].scatter([pos + .12] * len(dr), dr, color=colors[name], alpha=.55, marker="x")
        axes[0, 0].scatter([pos], [obs], color="black", s=70, marker="D")
    axes[0, 0].set_xticks([0, 1], ["base", "post-trained"]); axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("truth-preferred share at max alpha"); axes[0, 0].set_title("Observed (diamond) vs matched nulls")

    for name, rec in (("base", base), ("instruct", inst)):
        y = rec["native_test"]["labels"]["lie"]["yes_bias"]["selected"]
        axes[0, 1].plot(a := rec["native_test"]["alphas"], y["honesty"], color=colors[name], label=f"{name}: truth")
        axes[0, 1].plot(a, y["yes_push"], "--", color=colors[name], label=f"{name}: Yes push")
    axes[0, 1].axhline(0, color="#777", lw=1); axes[0, 1].set_title("Polarity decomposition")
    axes[0, 1].set_xlabel("steering coefficient"); axes[0, 1].set_ylabel("bits"); axes[0, 1].legend(fontsize=8)

    paired = summary["shared_comparison"]["labels"]["lie"]["alphas"]
    axes[1, 0].plot([x["alpha"] for x in paired], [x["base_mean_improvement"] for x in paired], "o-",
                    color=colors["base"], label="base")
    axes[1, 0].plot([x["alpha"] for x in paired], [x["instruct_mean_improvement"] for x in paired], "o-",
                    color=colors["instruct"], label="post-trained")
    axes[1, 0].set_title("Paired fixed-prompt mean improvement"); axes[1, 0].set_xlabel("steering coefficient")
    axes[1, 0].set_ylabel("bits toward truth"); axes[1, 0].legend()

    labels, vals = [], []
    for key, transfer in summary["cross_transfer"].items():
        if not transfer["ran"]:
            continue
        for mode, rec in transfer["modes"].items():
            labels.append(key.replace("_to_", "->") + "\n" + mode)
            vals.append(rec["labels"]["lie"]["arms"]["selected"]["share"][-1])
    axes[1, 1].bar(np.arange(len(vals)), vals, color="#5b8c85")
    axes[1, 1].set_xticks(np.arange(len(vals)), labels, rotation=35, ha="right", fontsize=8)
    axes[1, 1].set_ylim(0, 1); axes[1, 1].set_ylabel("truth-preferred share")
    axes[1, 1].set_title("Cross-transfer under four calibrations")
    fig.suptitle("Base vs post-training robustness diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(out / "base_vs_instruct_diagnostics.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict, out: Path) -> None:
    """Write a compact, data-derived Markdown report; never hard-code a conclusion."""
    lines = ["# Base vs post-trained honesty-circuit analysis", "",
             "All figures below are computed from this run. Rates use exact 95% binomial intervals; paired model "
             "comparisons use identical stored prompt IDs.", ""]
    for name in ("base", "instruct"):
        rec = summary[name]
        lie = rec["native_test"]["labels"]["lie"]
        selected = lie["arms"]["selected"]
        j = len(rec["native_test"]["alphas"]) - 1
        lines += [f"## {name}", "",
                  f"- Model: `{rec['model']}`",
                  f"- Selected circuit: {rec['selection']['n_selected']} heads using the "
                  f"`{rec['selection']['rule']}` IG aggregation rule; target met: {rec['selection']['met_target']}.",
                  f"- Held-out eligible lies: {lie['n']}.",
                  f"- Truth preferred at alpha={rec['native_test']['alphas'][j]}: "
                  f"{selected['share'][j]:.3f} (exact interval {selected['share_ci'][j][1]:.3f}-"
                  f"{selected['share_ci'][j][2]:.3f}).",
                  f"- Random-head empirical p: {lie['randomization_p']['random_heads']['share'][j]}; "
                  f"random-direction empirical p: {lie['randomization_p']['random_directions']['share'][j]}.", ""]
    sim = summary["direction_similarity"]
    lines += ["## Cross-checkpoint structure", ""]
    if sim["comparable"]:
        lines += [f"- Mean per-layer direction cosine: {sim['mean_layer_cosine']:.3f}.",
                  f"- Selected-head overlap: {sim['head_overlap']}; Jaccard: {sim['head_jaccard']:.3f}."]
    paired = summary["shared_comparison"]["labels"]["lie"]["alphas"][-1]
    lines += [f"- Fixed-prompt mean improvement difference (base minus post-trained) at max alpha: "
              f"{paired['mean_difference_base_minus_instruct']:.3f} bits; paired sign-flip p="
              f"{paired['paired_signflip_p']:.4f}; McNemar p={paired['mcnemar_p']:.4f}.", "",
              "## Interpretation guardrails", "",
              "Cross-transfer is reported under raw, global-norm, per-head-norm, and selection-effect calibration. "
              "Treat an apparent asymmetry as robust only when it persists across calibrations. The Yes-push and "
              "truth-directed components are reported separately in JSON and the diagnostics figure. Model-specific "
              "eligible-item rates and fixed-prompt paired results answer different questions and should both be shown.", ""]
    (out / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


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
    p.add_argument("--random", type=int, default=19,
                   help="Matched random head sets and random directions for native tests")
    p.add_argument("--cross-random", type=int, default=8,
                   help="Matched null draws per cross-transfer calibration mode")
    p.add_argument("--extensions", default="",
                   help="Comma-separated expensive stages: persona,audit,ablation,sycophancy, or all")
    p.add_argument("--extensions-only", action="store_true",
                   help="Reuse an existing core JSON/circuits and run only the requested extension stages")
    p.add_argument("--persona-items", default="data/items_persona.json")
    p.add_argument("--persona-random", type=int, default=39)
    p.add_argument("--audit-items", default="data/items_insider_audit.json")
    p.add_argument("--controls", default="data/controls.json")
    a = p.parse_args(); a.alphas = [float(x) for x in a.alphas.split(",")]
    a.extensions = [x.strip() for x in a.extensions.split(",") if x.strip()]
    unknown = set(a.extensions) - {"all", "persona", "audit", "ablation", "sycophancy"}
    if unknown:
        p.error(f"unknown --extensions: {sorted(unknown)}")
    torch.manual_seed(a.seed)
    root = Path(__file__).resolve().parent.parent
    for key in ("items", "persona_items", "audit_items", "controls"):
        value = Path(getattr(a, key))
        setattr(a, key, str((value if value.is_absolute() else root / value).resolve()))
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    items = load_items(a.items)
    if a.extensions_only:
        combined = out / "base_vs_instruct.json"
        if not combined.exists():
            p.error("--extensions-only requires an existing base_vs_instruct.json in --out")
        result = json.loads(combined.read_text(encoding="utf-8"))
        extensions = {"base": run_extensions("base", a.base_model, result["base"], a, root, out),
                      "instruct": run_extensions("instruct", a.instruct_model, result["instruct"], a, root, out)}
        result.setdefault("extensions", {}).update({k: v for k, v in extensions.items() if v})
        dump_json(combined, result)
        print(f"Updated extension outputs in {combined}", flush=True)
        return
    base = analyze_checkpoint("base", a.base_model, items, a, out)
    inst = analyze_checkpoint("instruct", a.instruct_model, items, a, out)
    if base["tokenizer"] != inst["tokenizer"]:
        raise RuntimeError("Checkpoint tokenizers differ on the stored answer/prefix IDs. Refusing an invalid comparison.")
    base_state, inst_state = out / "base_circuit.pt", out / "instruct_circuit.pt"
    source_base = {"name": "base", "circuit": base_state, "tokenizer": base["tokenizer"]}
    source_inst = {"name": "instruct", "circuit": inst_state, "tokenizer": inst["tokenizer"]}
    cross = {"base_to_instruct": cross_transfer("instruct", a.instruct_model, source_base, inst_state, items, a),
             "instruct_to_base": cross_transfer("base", a.base_model, source_inst, base_state, items, a)}
    result = {"config": {k: getattr(a, k) for k in ("base_model", "instruct_model", "family", "span", "bins", "alpha",
              "alphas", "target_share", "max_share", "ig_steps", "random", "cross_random", "seed")},
              "base": base, "instruct": inst, "shared_comparison": shared_comparison(base, inst, a),
              "direction_similarity": direction_similarity(base_state, inst_state), "cross_transfer": cross}
    dump_json(out / "base_vs_instruct.json", result)
    plot(result, out)
    write_report(result, out)
    extensions = {"base": run_extensions("base", a.base_model, base, a, root, out),
                  "instruct": run_extensions("instruct", a.instruct_model, inst, a, root, out)}
    if any(extensions.values()):
        result["extensions"] = extensions
        dump_json(out / "base_vs_instruct.json", result)
    print(f"Wrote JSON, two PNG figures, and {out / 'ANALYSIS.md'}", flush=True)


if __name__ == "__main__":
    main()
