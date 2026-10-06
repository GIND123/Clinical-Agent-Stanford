"""`deferdx` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import apply_overrides, load_config, load_yaml


def _base(args) -> dict:
    return load_config(args.config)


def _catalog_env(cfg: dict, args=None):
    from .env import EnvConfig, TestCatalog

    env = dict(cfg.get("env", {}))
    if args is not None and getattr(args, "no_defer", False):
        env["allow_defer"] = False
    if args is not None and getattr(args, "closed_world", False):
        env["open_world"] = False
    return TestCatalog.from_yaml(cfg["test_catalog"]), EnvConfig.from_dict(env)


def _severity(cfg: dict):
    from .rewards import SeverityMatrix

    path = cfg.get("reward", {}).get("severity_matrix")
    return SeverityMatrix.from_yaml(path) if path else SeverityMatrix.uniform()


def _load_many(paths: list[str]):
    from .data import load_cases

    cases = []
    for p in paths:
        cases.extend(load_cases(p))
    return cases


# ---- data ---------------------------------------------------------------------------


def cmd_data_synth(args):
    from .data.io import assign_splits, save_cases, write_splits
    from .data.synthetic import synth_cases

    cases = synth_cases(args.n, args.n_other, args.seed)
    out = Path(args.out)
    in_set = [c for c in cases if c.label != "other"]
    other = [c for c in cases if c.label == "other"]
    counts = write_splits(out, in_set, assign_splits(in_set, seed=args.seed))
    save_cases(out / "other.jsonl", other)
    print(f"synthetic cases -> {out}: {counts}, other={len(other)}  (SYNTHETIC: never report results on these)")


def cmd_data_inspect(args):
    from .data.cdm_loader import inspect_cdm

    print(inspect_cdm(args.cdm_dir))


def cmd_data_build_cdm(args):
    from collections import Counter

    from .data.cdm_loader import _find, load_cdm, load_pathology_ids
    from .data.io import assign_splits, load_split_file, save_cases, write_splits
    from .data.splits import lacdm_split, split_counts, stratified_split, subject_leakage

    fmt = load_yaml(args.format)
    icd = load_yaml(args.icd)["cdm_conditions"] if args.icd else None
    cases = load_cdm(args.cdm_dir, fmt, icd)
    out = Path(args.out)
    save_cases(out / "all.jsonl", cases)
    if args.split_file:
        splits = load_split_file(args.split_file)
    elif args.split == "lacdm":
        ids_path = _find(Path(args.cdm_dir), fmt.get("pathology_ids_file", "pathology_ids.json"))
        if ids_path is None:
            raise SystemExit("--split lacdm needs pathology_ids.json (CDM v1.1); use --split stratified instead")
        splits = lacdm_split(load_pathology_ids(ids_path))
    elif args.split == "stratified":
        splits = stratified_split([c.case_id for c in cases], [c.label for c in cases], tuple(args.ratios), args.seed)
    else:
        splits = assign_splits(cases, tuple(args.ratios), seed=args.seed)
    missing = sum(c.case_id not in splits for c in cases)
    if missing:
        print(f"WARNING: {missing} cases have no split assignment and were left out of train/val/test")
    leaks = subject_leakage(splits, {c.case_id: c.subject_id for c in cases})
    if leaks:
        print(f"NOTE: {leaks} subjects have admissions in more than one split (admission-level split)")
    counts = write_splits(out, cases, splits)
    print(f"{len(cases)} cases {dict(Counter(c.label for c in cases))} -> {out} {counts}")
    print("per split:", split_counts(splits, {c.case_id: c.label for c in cases}))


def cmd_data_build_ow(args):
    from collections import Counter

    from .data.io import load_cases, save_cases
    from .data.openworld import build_openworld

    icd = load_yaml(args.icd)
    exclude = {c.case_id for c in load_cases(args.exclude_cases)} if args.exclude_cases else set()
    cases = build_openworld(args.mimic_dir, args.note_dir, icd, target_n=args.n, seed=args.seed,
                            exclude_case_ids=exclude, include_controls=args.controls,
                            controls_per_label=args.controls_per_label)
    other = [c for c in cases if c.label == "other"]
    controls = [c for c in cases if c.label != "other"]
    out = Path(args.out)
    save_cases(out / "other.jsonl", other)
    if controls:
        save_cases(out / "controls.jsonl", controls)
    leaks = sum(c.meta.get("possible_label_leak", False) for c in cases)
    print(f"OTHER: {len(other)} {dict(Counter(c.meta['group'] for c in other))}; controls: {len(controls)}; "
          f"possible label leaks flagged: {leaks} -> {out}")


def cmd_data_cohorts(args):
    from .data.cohorts import build_cohorts, load_cohort_sources, save_cohorts

    train, val, test, other, controls, attached = load_cohort_sources(args.cdm_dir, args.openworld_dir,
                                                                      args.admissions)
    co = build_cohorts(train, val, test, other, controls, unseen_groups=args.unseen_groups, n_dev=args.n_dev,
                       other_eval_frac=args.other_eval_frac, controls_eval_frac=args.controls_eval_frac,
                       seed=args.seed)
    counts = save_cohorts(co, args.out)
    print(f"subject_id attached from admissions: {attached}")
    for k, v in co.report.items():
        if k != "counts":
            print(f"  {k}: {v}")
    print(f"-> {args.out} {counts}")


def cmd_data_leakage(args):
    from .data.leakage import source_classifier

    a, b = _load_many(args.a), _load_many(args.b)
    if args.label:
        a = [c for c in a if c.label == args.label]
        b = [c for c in b if c.label == args.label]
    rep = source_classifier(a, b, tuple(args.fields), folds=args.folds, seed=args.seed)
    print(f"A={rep.n_a} cases vs B={rep.n_b} cases; source-classifier AUROC {rep.auroc_mean:.3f} +/- {rep.auroc_std:.3f}")
    print(rep.verdict())
    print("features pointing to A:", rep.top_features_a)
    print("features pointing to B:", rep.top_features_b)


def cmd_data_coverage(args):
    from collections import Counter

    cfg = _base(args)
    catalog, _ = _catalog_env(cfg)
    cases = _load_many(args.cases)
    avail = Counter()
    for c in cases:
        avail.update(catalog.available_tests(c))
    print(f"{len(cases)} cases. Fraction of cases where each catalog test has a result:")
    for key in catalog.tests:
        print(f"  {key:16s} {avail[key] / max(1, len(cases)):.3f}")
    unreached = catalog.unreached_lab_names(cases)
    print(f"\nLab names never reachable through the catalog ({len(unreached)}), top {args.top}:")
    for name, n in list(unreached.items())[: args.top]:
        print(f"  {n:6d}  {name}")


# ---- rollouts / evaluation -------------------------------------------------------------


def _make_policy(args, cfg):
    kw = cfg.get("chat_template_kwargs", {"enable_thinking": True})
    if args.policy == "oracle":
        from .policy import OraclePolicy

        return OraclePolicy(seed=args.seed)
    if args.policy == "random":
        from .policy import RandomPolicy

        return RandomPolicy(seed=args.seed)
    if args.policy == "hf":
        from .policy.hf import HFPolicy

        return HFPolicy.from_pretrained(args.model, args.adapter, max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature, do_sample=args.temperature > 0,
                                        top_p=args.top_p, top_k=args.top_k,
                                        batch_size=args.batch_size, chat_template_kwargs=kw)
    if args.policy == "vllm":
        from .policy.vllm_policy import VLLMPolicy

        extra = {"max_model_len": args.max_model_len} if getattr(args, "max_model_len", None) else {}
        return VLLMPolicy(args.model, args.adapter, max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                          chat_template_kwargs=kw, seed=args.seed, **extra)
    raise ValueError(args.policy)


def cmd_rollout(args):
    from .data.io import write_jsonl
    from .rollout import run_episodes

    cfg = _base(args)
    catalog, env_cfg = _catalog_env(cfg, args)
    cases = _load_many(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    policy = _make_policy(args, cfg)
    rollouts = run_episodes(policy, catalog, env_cfg, cases, n_samples=args.samples,
                            on_turn=lambda t, n: print(f"  turn {t}: {n} active", file=sys.stderr))
    n = write_jsonl(args.out, (ro.to_dict() for ro in rollouts))
    print(f"{n} rollouts -> {args.out}")


def _results(path):
    from .data.io import read_jsonl
    from .env import EpisodeResult

    return [EpisodeResult.from_dict(r["result"]) for r in read_jsonl(path)]


def cmd_evaluate(args):
    from .eval import format_report, summarize

    cfg = _base(args)
    results = _results(args.rollouts)
    cf = None
    if args.counterfactual:
        cf = {r.case_id: r.correct for r in _results(args.counterfactual)}
    rep = summarize(results, _severity(cfg), cf, confident_threshold=args.confident_threshold)
    print(format_report(rep))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")


def cmd_baseline(args):
    from .baselines import apply_threshold, sgr_threshold, threshold_for_coverage, threshold_for_risk
    from .eval import format_report, summarize

    cfg = _base(args)
    val, test = _results(args.val), _results(args.test)
    if args.method == "coverage":
        thr = threshold_for_coverage(val, args.target)
    elif args.method == "risk":
        thr = threshold_for_risk(val, args.target)
    else:
        thr = sgr_threshold(val, args.target, args.delta)
    print(f"{args.method} threshold (fit on val) = {thr}")
    rep = summarize(apply_threshold(test, thr), _severity(cfg))
    rep["threshold"] = {"method": args.method, "target": args.target, "value": thr}
    print(format_report(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")


# ---- SFT data / training -----------------------------------------------------------------


def cmd_sft_oracle(args):
    from .data.io import write_jsonl
    from .training.sft_data import oracle_sft

    cfg = _base(args)
    catalog, env_cfg = _catalog_env(cfg, args)
    rows = oracle_sft(_load_many(args.cases), catalog, env_cfg, args.defer_fraction, args.seed)
    print(f"{write_jsonl(args.out, rows)} oracle trajectories -> {args.out}")


def cmd_sft_rollouts(args):
    from collections import Counter

    from .data.io import read_jsonl, write_jsonl
    from .training.sft_data import rollouts_sft

    rows = rollouts_sft(read_jsonl(args.rollouts), args.tau, args.max_correct_per_case, args.max_defer_fraction,
                        args.seed)
    write_jsonl(args.out, rows)
    print(f"{len(rows)} trajectories {dict(Counter(r['kind'] for r in rows))} -> {args.out}")


def cmd_train_sft(args):
    from .training.sft import train_sft

    print(f"saved -> {train_sft(apply_overrides(load_config(args.config), args.set))}")


def cmd_train_grpo(args):
    from .training.grpo import train_grpo

    print(f"saved -> {train_grpo(apply_overrides(load_config(args.config), args.set))}")


def cmd_train_grpo_vllm(args):
    from .training.grpo_vllm import train_grpo_vllm

    print(f"saved -> {train_grpo_vllm(apply_overrides(load_config(args.config), args.set))}")


def cmd_crossover(args):
    import numpy as np

    from .rewards import RewardConfig, crossover_p_hat

    cfg = _base(args)
    catalog, _ = _catalog_env(cfg)
    rcfg = RewardConfig.from_dict(cfg.get("reward"), _severity(cfg), catalog.total_test_cost)
    print(f"cost_scale={rcfg.cost_scale:.3g} per $ (CT = {rcfg.cost_scale * catalog.tests['ct_abdomen'].cost:.3f} reward)")
    print("tau   -> reward-induced crossover p_hat (defer below, commit above)")
    for tau in np.round(np.arange(0.6, 1.0001, 0.05), 2):
        rcfg.tau = float(tau)
        q = crossover_p_hat(rcfg)
        print(f"{tau:.2f}  -> {q:.3f}" if q is not None else f"{tau:.2f}  -> never commits")


# ---- parser ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deferdx", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_config(sp, default="configs/base.yaml"):
        sp.add_argument("--config", default=default)
        return sp

    data = sub.add_parser("data", help="build / inspect datasets").add_subparsers(dest="sub", required=True)
    sp = data.add_parser("synth", help="synthetic cases for pipeline tests")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--n", type=int, default=400)
    sp.add_argument("--n-other", type=int, default=80)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_data_synth)

    sp = data.add_parser("inspect-cdm", help="list tables/columns of a MIMIC-IV-Ext-CDM download")
    sp.add_argument("--cdm-dir", required=True)
    sp.set_defaults(fn=cmd_data_inspect)

    sp = data.add_parser("build-cdm", help="MIMIC-IV-Ext-CDM -> canonical JSONL + splits")
    sp.add_argument("--cdm-dir", required=True)
    sp.add_argument("--format", default="configs/cdm_format.yaml")
    sp.add_argument("--icd", default="configs/openworld_icd.yaml")
    sp.add_argument("--split", choices=["lacdm", "stratified", "hash"], default="lacdm",
                    help="lacdm: exact LA-CDM 80/10/10 (default, comparable to LA-CDM/ReAct numbers); "
                         "stratified/hash use --ratios (e.g. 0.7 0.1 0.2 as LDTL reports)")
    sp.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    sp.add_argument("--split-file", help="explicit {case_id: split} file; overrides --split")
    sp.add_argument("--out", default="data/cdm")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_data_build_cdm)

    sp = data.add_parser("build-openworld", help="MIMIC-IV -> MIMIC-CDM-OW (label OTHER)")
    sp.add_argument("--mimic-dir", required=True, help="MIMIC-IV v2.2 root (contains hosp/)")
    sp.add_argument("--note-dir", required=True, help="MIMIC-IV-Note root (contains note/)")
    sp.add_argument("--icd", default="configs/openworld_icd.yaml")
    sp.add_argument("--exclude-cases", help="CDM all.jsonl; these hadm_ids are excluded")
    sp.add_argument("--n", type=int, default=2400,
                    help="OTHER cases sampled BEFORE the history-leak and inclusion drops; 2400 gives the "
                         "documented 713-case set (800 gives only 259)")
    sp.add_argument("--controls", action="store_true", help="also extract same-pipeline in-set controls")
    sp.add_argument("--controls-per-label", type=int, default=100)
    sp.add_argument("--out", default="data/openworld")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_data_build_ow)

    sp = data.add_parser("cohorts", help="patient-disjoint RL-train / dev / evaluation sets (CDM + open world)")
    sp.add_argument("--cdm-dir", default="data/cdm")
    sp.add_argument("--openworld-dir", default="data/openworld_xl")
    sp.add_argument("--admissions", default="data/physionet/mimiciv/2.2/hosp/admissions.csv.gz",
                    help="MIMIC-IV admissions table, to attach subject_id to CDM cases")
    sp.add_argument("--unseen-groups", nargs="*", default=None,
                    help="OTHER groups kept out of training (default: the time-critical groups)")
    sp.add_argument("--n-dev", type=int, default=160)
    sp.add_argument("--other-eval-frac", type=float, default=0.4)
    sp.add_argument("--controls-eval-frac", type=float, default=0.4)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="data/cohorts")
    sp.set_defaults(fn=cmd_data_cohorts)

    sp = data.add_parser("leakage-check", help="can a text classifier tell two case sources apart?")
    sp.add_argument("--a", nargs="+", required=True, help="e.g. data/openworld/controls.jsonl")
    sp.add_argument("--b", nargs="+", required=True, help="e.g. data/cdm/all.jsonl")
    sp.add_argument("--label", help="restrict both sides to one label (recommended)")
    sp.add_argument("--fields", nargs="+", default=["hpi", "physical_exam", "imaging"])
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_data_leakage)

    sp = with_config(data.add_parser("coverage", help="catalog coverage audit on a case file"))
    sp.add_argument("--cases", nargs="+", required=True)
    sp.add_argument("--top", type=int, default=30)
    sp.set_defaults(fn=cmd_data_coverage)

    sp = with_config(sub.add_parser("rollout", help="run a policy in the environment"))
    sp.add_argument("--cases", nargs="+", required=True)
    sp.add_argument("--policy", choices=["oracle", "random", "hf", "vllm"], required=True)
    sp.add_argument("--model")
    sp.add_argument("--adapter")
    sp.add_argument("--samples", type=int, default=1)
    # Defaults = Qwen3 thinking-mode recommendation (model card: do NOT decode greedily).
    # Report mean +/- sd over several --seed values rather than a single sampled run.
    sp.add_argument("--temperature", type=float, default=0.6)
    sp.add_argument("--top-p", type=float, default=0.95)
    sp.add_argument("--top-k", type=int, default=20)
    sp.add_argument("--max-new-tokens", type=int, default=1024)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--max-model-len", type=int, help="vLLM context cap; lower it when weights leave little KV-cache room")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--no-defer", action="store_true", help="forced-choice control")
    sp.add_argument("--closed-world", action="store_true", help="do not offer OTHER")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_rollout)

    sp = with_config(sub.add_parser("evaluate", help="metric report for a rollout file"))
    sp.add_argument("--rollouts", required=True)
    sp.add_argument("--counterfactual", help="no-defer rollouts of the same cases (deferral P/R target)")
    sp.add_argument("--confident-threshold", type=float, default=0.8)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_evaluate)

    sp = with_config(sub.add_parser("baseline", help="post-hoc threshold / SGR on no-defer rollouts"))
    sp.add_argument("--method", choices=["coverage", "risk", "sgr"], required=True)
    sp.add_argument("--val", required=True)
    sp.add_argument("--test", required=True)
    sp.add_argument("--target", type=float, required=True, help="coverage or selective risk")
    sp.add_argument("--delta", type=float, default=0.05)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_baseline)

    sft = sub.add_parser("sft-data", help="build SFT trajectories").add_subparsers(dest="sub", required=True)
    sp = with_config(sft.add_parser("oracle"))
    sp.add_argument("--cases", nargs="+", required=True)
    sp.add_argument("--defer-fraction", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_sft_oracle)
    sp = sft.add_parser("rollouts", help="STaR rejection sampling + deferral exemplars")
    sp.add_argument("--rollouts", required=True)
    sp.add_argument("--tau", type=float, default=0.85)
    sp.add_argument("--max-correct-per-case", type=int, default=2)
    sp.add_argument("--max-defer-fraction", type=float, default=0.3)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_sft_rollouts)

    train = sub.add_parser("train", help="SFT / GRPO").add_subparsers(dest="sub", required=True)
    for name, fn, default in (("sft", cmd_train_sft, "configs/sft.yaml"), ("grpo", cmd_train_grpo, "configs/grpo.yaml"),
                              ("grpo-vllm", cmd_train_grpo_vllm, "configs/grpo_deferdx.yaml")):
        sp = with_config(train.add_parser(name), default)
        sp.add_argument("--set", nargs="*", help="overrides, e.g. steps=10 group_size=8")
        sp.set_defaults(fn=fn)

    sp = with_config(sub.add_parser("crossover", help="effective deferral threshold induced by the reward"))
    sp.set_defaults(fn=cmd_crossover)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
