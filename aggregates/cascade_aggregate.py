# cascade_aggregate.py — per-scenario infection probabilities (only round/agent/prob)
# Usage:
#   python3 cascade_aggregate.py --csqa-jsonl ./dataset/csqa.jsonl
# Optional:
#   --root ./output/gpt-4o-mini/csqa         (where your scenario folders live)
#   --max-round 10                           (keep rounds 0..max_round)
#   --model gpt-4o-mini --dataset csqa       (kept for path shape defaults)

import argparse, os, json, re, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# -------- Regexes / parsers --------
ANSWER_RE   = re.compile(r"<ANSWER>:\s*([A-Z])\b", re.IGNORECASE)
UPD_ANS_RE  = re.compile(r"<UPDATED_ANSWER>:\s*([A-Z])\b", re.IGNORECASE)
IDX_RE      = re.compile(r"_idx-([0-9\-]+)[^/]*\.output$")
META_RE     = re.compile(r"^csqa_(?P<graph>[a-z_]+)_(?P<n>\d+)_", re.I)

def parse_line_to_obj(line: str):
    try:
        return json.loads(line)
    except Exception:
        return eval(line)

def extract_agent_answers(dialogue):
    """Collect option letters from assistant messages across rounds."""
    out = []
    for msg in dialogue:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        ans = None
        if isinstance(content, dict):
            ans = content.get("answer") or content.get("updated_answer")
        else:
            text = str(content)
            m = UPD_ANS_RE.search(text) or ANSWER_RE.search(text)
            if m:
                ans = m.group(1).upper()
        if ans:
            a = str(ans).strip()
            if a and a[0] in "ABCDE":
                out.append(a[0])
    return out

def load_gold_map(csqa_jsonl: str):
    gold = {}
    with open(csqa_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = eval(line.strip())
            except Exception:
                continue
            gid = item.get("id")
            g = str(item.get("answerKey")).strip().upper()
            if gid and g:
                gold[gid] = g
    return gold

def parse_graph_agents_idxs_from_name(fname: str):
    """
    Return (graph, n_agents, idxs:list[int]) from filename like:
      csqa_pure_star_6_2_idx-0-2.output, csqa_complete_6_0.output, etc.
    Handles extra suffix after idx-... (e.g., _comm-vote_only).
    """
    graph, n_agents = None, None
    mm = META_RE.match(fname)
    if mm:
        graph = mm.group("graph")
        try:
            n_agents = int(mm.group("n"))
        except Exception:
            n_agents = None

    m = IDX_RE.search(fname)
    if not m:
        return graph, n_agents, []
    idxs = [int(x) for x in m.group(1).split("-") if x.strip() != ""]
    return graph, n_agents, idxs

def scenario_label(graph: str, idxs: list[int]) -> str:
    """Human-friendly scenario name for the CSV filename."""
    if not idxs:
        return "baseline"
    if graph in {"star", "pure_star"}:
        if len(idxs) == 1:
            return "hub" if idxs[0] == 0 else f"leaf{idxs[0]}"
        if len(idxs) == 2:
            if 0 in idxs:
                other = [x for x in idxs if x != 0][0]
                return f"hub+leaf{other}"
            return f"leaves{idxs[0]}-{idxs[1]}"
        return f"{len(idxs)}-attackers"
    # complete / others: no 'hub/leaf' semantics
    return "idx-" + "-".join(map(str, idxs))

# -------- New: detect scenario modifiers from folder name --------
MOD_PATTERNS = {
    "voteonly": re.compile(r"vote[\s\-_]*only", re.I),
    "trust":    re.compile(r"\btrust\b", re.I),
    "htrust":    re.compile(r"\bhtrust\b", re.I),
    "ltrust":    re.compile(r"\bltrust\b", re.I),
}

def detect_modifier_from_path(fp: str) -> str:
    """
    Looks at the immediate scenario folder (the parent of the seed folder).
    Returns a normalized modifier string like 'voteonly', 'trust', or 'voteonly_trust', or ''.
    """
    # path shape: .../<scenario_folder>/<seed>/<file.output>
    p = Path(fp)
    scenario_folder = p.parent.parent.name  # up two: seed -> scenario
    name = scenario_folder.lower()
    mods = []
    for label, pat in MOD_PATTERNS.items():
        if pat.search(name):
            mods.append(label)
    return "_".join(sorted(set(mods))) if mods else ""

# -------- Core aggregation --------
def aggregate_scenario(files: list[str], gold_map: dict, idxs: list[int], max_round: int | None):
    """
    Return tidy DataFrame with columns: round, agent, infected_prob
    Aggregated across *all tasks & seeds & files* in this scenario group.
    """
    rows = []
    for path in files:
        seed = os.path.basename(os.path.dirname(path))  # folder named like 1,3,5,...
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = parse_line_to_obj(s)
                task_id = obj.get("task_id")
                gold = gold_map.get(task_id)
                if gold is None:
                    # skip tasks not found in gold
                    continue

                # Collect answers per agent/round
                agent_to_answers = {}
                for k, v in obj.items():
                    if not str(k).startswith("Agent_"):
                        continue
                    try:
                        agent_idx = int(k.split("_")[1])
                    except Exception:
                        continue
                    agent_to_answers[agent_idx] = extract_agent_answers(v)

                # Emit per (agent, round) wrong flag
                for agent_idx, ans_seq in agent_to_answers.items():
                    T = len(ans_seq)
                    for r in range(T):
                        if (max_round is not None) and (r > max_round):
                            break
                        ans = ans_seq[r]
                        if r == 0:
                            # Round 0 override: only attackers are 'infected'
                            infected = (agent_idx in idxs)
                        else:
                            infected = (ans.upper() != gold.upper())
                        rows.append({
                            "seed": int(seed) if seed.isdigit() else seed,
                            "agent": agent_idx,
                            "round": r,
                            "infected": 1.0 if infected else 0.0,
                        })

    if not rows:
        return pd.DataFrame(columns=["round","agent","infected_prob"])

    df = pd.DataFrame(rows)
    agg = (df.groupby(["round","agent"])["infected"]
             .mean()
             .rename("infected_prob")
             .reset_index()
             .sort_values(["round","agent"]))
    return agg

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Build per-scenario infection probability CSVs.")
    ap.add_argument("--root", default="./output/gpt-4o-mini/csqa",
                    help="Root directory that contains scenario folders with seed subfolders.")
    ap.add_argument("--csqa-jsonl", required=True, help="Path to csqa.jsonl (for gold answers).")
    ap.add_argument("--max-round", type=int, default=10,
                    help="Keep rounds 0..max_round (inclusive). Use -1 to keep all).")
    args = ap.parse_args()

    gold_map = load_gold_map(args.csqa_jsonl)
    out_dir = Path("./aggregates")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all .output files under root/**/*
    pattern = str(Path(args.root) / "**" / "*.output")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"No .output files found under {args.root}")

    # Group by (graph, n_agents, attacker idx tuple, modifier)
    groups = defaultdict(list)
    meta_for_key = {}

    for fp in files:
        fname = os.path.basename(fp)
        graph, n_agents, idxs = parse_graph_agents_idxs_from_name(fname)
        if graph is None or n_agents is None:
            # skip unknown shapes
            continue
        modifier = detect_modifier_from_path(fp)  # NEW
        key = (graph, n_agents, tuple(sorted(idxs)), modifier)
        groups[key].append(fp)
        meta_for_key[key] = (graph, n_agents, idxs, modifier)

    if not groups:
        raise SystemExit("No recognizable files (graph/n) found.")

    kept_round = None if args.max_round is None or args.max_round < 0 else int(args.max_round)

    for key, flist in sorted(groups.items()):
        graph, n_agents, idxs, modifier = meta_for_key[key]
        scen = scenario_label(graph, idxs)
        mod_suffix = f"_{modifier}" if modifier else ""
        out_csv = out_dir / f"infect_prob_{graph}_{n_agents}a_{scen}{mod_suffix}.csv"

        # Build & save
        agg = aggregate_scenario(flist, gold_map, idxs, kept_round)
        # Only the three columns requested:
        agg = agg[["round","agent","infected_prob"]]
        agg.to_csv(out_csv, index=False)
        print(f"[OK] {scen:<16} -> {out_csv}  (rows={len(agg)})")

if __name__ == "__main__":
    main()
