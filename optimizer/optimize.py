"""
Optimizer Harness
Uses eval scores to iteratively improve the auditor agent's system prompt.

Strategy:
  1. Run baseline evals → get S(t,C) scores including efficiency + cost
  2. Identify weakest dimensions (accuracy sub-dims, efficiency, cost)
  3. Use Claude Haiku to generate an improved system prompt targeting those
  4. Re-run evals with the new prompt
  5. Keep the best; repeat for N rounds
  6. Print before/after table with Δ for every metric
"""

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ResultMessage,
)

from agent.auditor import AUDITOR_SYSTEM_PROMPT
from eval.harness import run_evals, EvalResult, W_ACC, W_EFF, W_COST


OPTIMIZER_MODEL = "claude-haiku-4-5-20251001"   # cheap model keeps optimizer costs low
MAX_ROUNDS  = 3
RESULTS_DIR = "results"


# ── Optimizer prompt builder ──────────────────────────────────────────────────

def build_optimizer_prompt(
    current_prompt: str,
    eval_results: list[EvalResult],
    round_num: int,
) -> str:
    lines = []
    for r in eval_results:
        lines.append(
            f"- {r.test_id}:\n"
            f"    S(t,C)={r.overall:.3f}  accuracy={r.accuracy:.3f}  "
            f"efficiency={r.efficiency:.3f}  cost_score={r.cost_score:.3f}\n"
            f"    accuracy breakdown → validity={r.report_validity:.2f}  "
            f"score_accuracy={r.score_accuracy:.2f}  "
            f"weakness_recall={r.weakness_recall:.2f}  "
            f"rewrite_quality={r.rewrite_quality:.2f}\n"
            f"    turns={r.num_turns}  cost_usd=${r.cost_usd:.4f}"
        )

    worst = []
    for r in eval_results:
        if r.weakness_recall < 0.7:
            worst.append(f"weakness_recall ({r.test_id}): {r.weakness_recall:.2f}")
        if r.rewrite_quality < 0.7:
            worst.append(f"rewrite_quality ({r.test_id}): {r.rewrite_quality:.2f}")
        if r.score_accuracy < 0.7:
            worst.append(f"score_accuracy ({r.test_id}): {r.score_accuracy:.2f}")
        if r.efficiency < 0.6:
            worst.append(f"efficiency ({r.test_id}): {r.efficiency:.2f}  — agent using too many turns")

    return f"""You are a prompt engineer improving an AI Agent Auditor's system prompt.
The auditor is scored with this formula:
  S(t,C) = {W_ACC}·Acc + {W_EFF}·Efficiency - {W_COST}·CostPenalty

ROUND: {round_num}

CURRENT SYSTEM PROMPT:
{current_prompt}

EVAL RESULTS:
{chr(10).join(lines)}

WEAKEST AREAS (priority targets):
{chr(10).join(worst) or "All metrics above threshold — fine-tune for marginal gains"}

YOUR TASK:
Rewrite the system prompt to improve the weakest areas above.

Specific guidance:
- To improve weakness_recall: instruct the auditor to EXPLICITLY check for hallucination
  risks, missing safety guardrails, authorisation gaps, destructive ops without confirmation,
  and vague tool descriptions
- To improve rewrite_quality: instruct the auditor's rewrites to include phrases like
  "never fabricate", "confirm before any destructive operation", "do not issue refunds
  without explicit verification", "escalate to a human"
- To improve score_accuracy: instruct the auditor to score safety/hallucination_risk LOW
  (0-4) for prompts that are clearly unsafe or encourage fabrication
- To improve efficiency: instruct the auditor to be concise — complete the audit in as
  few steps as possible (load spec → analyse → save report → done)

Keep all existing good behaviours. Only strengthen weak areas.

Return ONLY the new system prompt text. No explanation, no markdown fences."""


# ── Generate improved prompt via Claude Haiku ─────────────────────────────────

async def generate_improved_prompt(
    current_prompt: str,
    eval_results: list[EvalResult],
    round_num: int,
) -> str:
    optimizer_prompt = build_optimizer_prompt(current_prompt, eval_results, round_num)

    options = ClaudeAgentOptions(
        model=OPTIMIZER_MODEL,
        max_turns=1,
        permission_mode="acceptEdits",
    )

    new_prompt = ""
    async for message in query(prompt=optimizer_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    new_prompt += block.text
        elif isinstance(message, ResultMessage):
            print(f"  [optimizer] turns={message.num_turns} cost=${message.total_cost_usd:.4f}")

    return new_prompt.strip()


# ── Main optimizer loop ────────────────────────────────────────────────────────

async def run_optimizer():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    import agent.auditor as auditor_module

    history = []

    print("=" * 70)
    print("🚀 OPTIMIZER — Round 0 (baseline)")
    print("=" * 70)

    current_prompt = AUDITOR_SYSTEM_PROMPT
    baseline_results = await run_evals(results_dir=RESULTS_DIR, tag="round_0")

    def snapshot(rnd, results, prompt):
        return {
            "round":           rnd,
            "mean_overall":    round(sum(r.overall    for r in results) / len(results), 3),
            "mean_accuracy":   round(sum(r.accuracy   for r in results) / len(results), 3),
            "mean_efficiency": round(sum(r.efficiency for r in results) / len(results), 3),
            "mean_cost_usd":   round(sum(r.cost_usd   for r in results) / len(results), 4),
            "total_cost_usd":  round(sum(r.cost_usd   for r in results), 4),
            "prompt": prompt,
            "results_obj": results,
        }

    history.append(snapshot(0, baseline_results, current_prompt))
    best_mean    = history[0]["mean_overall"]
    best_prompt  = current_prompt
    round_results = baseline_results

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"🔧 OPTIMIZER — Round {rnd}: generating improved prompt via {OPTIMIZER_MODEL}...")
        print(f"{'='*70}")

        new_prompt = await generate_improved_prompt(current_prompt, round_results, rnd)

        if not new_prompt or len(new_prompt) < 100:
            print("⚠️  Optimizer returned empty/short prompt — skipping round")
            continue

        print(f"\n📝 New prompt preview ({len(new_prompt)} chars):")
        print(new_prompt[:400] + ("..." if len(new_prompt) > 400 else ""))

        auditor_module.AUDITOR_SYSTEM_PROMPT = new_prompt

        round_results = await run_evals(results_dir=RESULTS_DIR, tag=f"round_{rnd}")
        snap = snapshot(rnd, round_results, new_prompt)
        history.append(snap)

        delta = snap["mean_overall"] - best_mean
        print(f"\n📈 Round {rnd} S(t,C): {snap['mean_overall']:.3f}  "
              f"(Δ {delta:+.3f} vs best {best_mean:.3f})")

        if snap["mean_overall"] > best_mean:
            best_mean   = snap["mean_overall"]
            best_prompt = new_prompt
            current_prompt = new_prompt
            print("✅ Improvement! Keeping new prompt.")
        else:
            auditor_module.AUDITOR_SYSTEM_PROMPT = best_prompt
            current_prompt = best_prompt
            print("⬇️  No improvement — reverting to best.")

    # ── Final before/after report ──────────────────────────────────────────────

    before = history[0]
    best   = max(history, key=lambda h: h["mean_overall"])

    print("\n" + "=" * 70)
    print("🏁 OPTIMIZATION COMPLETE — BEFORE vs AFTER")
    print("=" * 70)
    print(f"\n{'Metric':<22} {'Round 0':>10} {'Best':>10} {'Δ':>10}")
    print("─" * 55)

    for key, label in [
        ("mean_overall",    "S(t,C) overall"),
        ("mean_accuracy",   "Accuracy"),
        ("mean_efficiency", "Efficiency"),
        ("mean_cost_usd",   "Avg cost (USD)"),
        ("total_cost_usd",  "Total cost (USD)"),
    ]:
        b0  = before[key]
        b1  = best[key]
        neg = key in ("mean_cost_usd", "total_cost_usd")
        delta = b1 - b0
        arrow = ("✅" if (delta < 0 if neg else delta > 0) else
                 "➡️" if delta == 0 else "⚠️")
        print(f"  {label:<20} {b0:>10.4f} {b1:>10.4f} {delta:>+10.4f}  {arrow}")

    pct = (best["mean_overall"] - before["mean_overall"]) / max(before["mean_overall"], 1e-9) * 100
    print(f"\n  Overall improvement: {pct:+.1f}%  (best at round {best['round']})")

    final_report = {
        "formula": f"S(t,C) = {W_ACC}*Acc + {W_EFF}*Eff - {W_COST}*(1-CostScore)",
        "baseline": {k: before[k] for k in before if k not in ("prompt", "results_obj")},
        "best":     {k: best[k]   for k in best   if k not in ("prompt", "results_obj")},
        "delta_overall": round(best["mean_overall"] - before["mean_overall"], 3),
        "pct_improvement": round(pct, 1),
        "best_round": best["round"],
        "history": [{k: h[k] for k in h if k not in ("prompt", "results_obj")} for h in history],
        "final_best_prompt": best_prompt,
    }
    out_path = Path(RESULTS_DIR) / "optimizer_report.json"
    out_path.write_text(json.dumps(final_report, indent=2))
    print(f"\n📁 Full optimizer report → {out_path}")

    # ── Also write baseline.json / optimized.json / comparison.md ───────────────
    # (these are the filenames referenced by README.md; regenerated each run so
    #  they always match the current code + current eval results)

    def results_to_json(tag: str, results: list[EvalResult]) -> dict:
        return {
            "tag": tag,
            "results": [
                {
                    "test_id":    r.test_id,
                    "overall":    r.overall,
                    "passed":     r.passed,
                    "accuracy":   round(r.accuracy, 3),
                    "efficiency": round(r.efficiency, 3),
                    "cost_score": round(r.cost_score, 3),
                    "num_turns":  r.num_turns,
                    "cost_usd":   round(r.cost_usd, 4),
                    "sub_dimensions": {
                        "report_validity": round(r.report_validity, 3),
                        "score_accuracy":  round(r.score_accuracy, 3),
                        "weakness_recall": round(r.weakness_recall, 3),
                        "rewrite_quality": round(r.rewrite_quality, 3),
                    },
                }
                for r in results
            ],
            "mean_overall":    round(sum(r.overall    for r in results) / len(results), 3),
            "mean_accuracy":   round(sum(r.accuracy   for r in results) / len(results), 3),
            "mean_efficiency": round(sum(r.efficiency for r in results) / len(results), 3),
            "total_cost_usd":  round(sum(r.cost_usd   for r in results), 4),
            "pass_rate":       round(sum(r.passed     for r in results) / len(results), 3),
        }

    baseline_json = results_to_json("baseline", baseline_results)

    optimized_json = results_to_json("optimized", best["results_obj"])

    (Path(RESULTS_DIR) / "baseline.json").write_text(json.dumps(baseline_json, indent=2))
    (Path(RESULTS_DIR) / "optimized.json").write_text(json.dumps(optimized_json, indent=2))

    md = []
    md.append("# Optimization Results\n")
    md.append("> Generated automatically by `python main.py optimize` — reflects the current code + most recent run.\n")
    md.append("## Summary\n")
    md.append("| Metric | Baseline Prompt | Optimized Prompt | Δ |")
    md.append("|--------|:--------------:|:---------------:|:---:|")
    d_overall = optimized_json["mean_overall"] - baseline_json["mean_overall"]
    d_acc = optimized_json["mean_accuracy"] - baseline_json["mean_accuracy"]
    pct_o = (d_overall / max(baseline_json["mean_overall"], 1e-9)) * 100
    md.append(f"| **Mean S(t,C)** | **{baseline_json['mean_overall']:.3f}** | **{optimized_json['mean_overall']:.3f}** | **{d_overall:+.3f} ({pct_o:+.1f}%)** |")
    md.append(f"| Mean Accuracy | {baseline_json['mean_accuracy']:.3f} | {optimized_json['mean_accuracy']:.3f} | {d_acc:+.3f} |")
    md.append(f"| Pass Rate | {baseline_json['pass_rate']:.0%} | {optimized_json['pass_rate']:.0%} | — |")
    md.append(f"| Total API Cost | ${baseline_json['total_cost_usd']:.4f} | ${optimized_json['total_cost_usd']:.4f} | — |\n")

    md.append("## Per-Test Breakdown\n")
    md.append("| Test Case | Baseline S(t,C) | Optimized S(t,C) | Δ |")
    md.append("|-----------|:-----------:|:------------:|:---:|")
    opt_by_id = {r["test_id"]: r for r in optimized_json["results"]}
    for r in baseline_json["results"]:
        tid = r["test_id"]
        b = r["overall"]
        o = opt_by_id.get(tid, {}).get("overall", float("nan"))
        delta = o - b
        mark = "✅" if delta > 0 else ("➡️" if delta == 0 else "⚠️")
        md.append(f"| `{tid}` | {b:.3f} | {o:.3f} | {delta:+.3f} {mark} |")

    md.append("\n## Files\n")
    md.append("```")
    md.append("results/")
    md.append("├── baseline.json          ← full baseline eval (round 0)")
    md.append("├── optimized.json         ← full eval for best-performing prompt")
    md.append("├── optimizer_report.json  ← full history across all rounds")
    md.append("├── comparison.md          ← this file")
    md.append("└── traces/                ← per-test traces with full agent reports")
    md.append("```")

    (Path(RESULTS_DIR) / "comparison.md").write_text("\n".join(md) + "\n")
    print(f"\n📁 baseline.json / optimized.json / comparison.md written to {RESULTS_DIR}/")

    return final_report


if __name__ == "__main__":
    asyncio.run(run_optimizer())
