# SDC: Semantic-Distance Checkpointing

Code and experiments for the paper on detecting goal drift in LLM-based multi-agent systems.

**Authors:** Anabiya Fatima, Menal Shahid. Department of Artificial Intelligence and Data Science, FAST-NUCES, Islamabad.

In a multi-agent LLM pipeline (Planner, Executor with a tool, Verifier), one small deviation from the user's goal, such as a substituted sub-task or an invented fact, is passed from agent to agent and grows. This is **goal drift**. **Semantic-Distance Checkpointing (SDC)** is a training-free monitor that scores every Planner and tool output by its cosine distance to the embedded user goal. When the score goes above a threshold calibrated on correct runs, it sends the Planner back to the original goal. The paper reports both the cost benefits and the limits of this idea.

## How SDC works

1. **Embed the goal.** The user question is embedded with a small sentence encoder (all-MiniLM-L6-v2).
2. **Score every step.** After the Planner and after each tool call, the output is embedded and its cosine distance to the goal is computed.
3. **Thresholds.** One threshold per step type (plan, tool) is set at a percentile of the scores of correct clean runs (85, 90 or 95, picked on a calibration split).
4. **Re-plan.** If the score exceeds the threshold, the Planner is re-prompted with the original goal (at most 2 re-plans per run).

## Main results

Qwen2.5-3B-Instruct, HotpotQA and GSM8K, 28 test tasks each, two drift seeds, one drift event injected per run.

| | HotpotQA | GSM8K |
|---|---|---|
| AUROC of the raw score for locating the injected step | 0.784 | 0.588 |
| False alarms on clean runs, SDC | 0.464 | 0.143 |
| False alarms on clean runs, per-step LLM judge | 1.000 | 0.964 |
| Extra tokens, SDC / LLM judge | 223 / 968 | 95 / 1,237 |
| Success under drift, SDC vs plain pipeline | 0.339 vs 0.375 | 0.339 vs 0.357 |

- SDC is **4.3 to 13.1 times cheaper** than a per-step LLM judge and raises far fewer false alarms.
- It detects goal substitution best (detect rate 1.00 on HotpotQA).
- It did **not** significantly improve task success.
- On 184 real Who&When failure logs it located the decisive step in only 1.1% of cases, below random guessing (9.0%).

All numbers are from the paper. The sample is small, so confidence intervals are wide.

## Repository contents

| File | Description |
|---|---|
| `notebook.ipynb` | The full experiment: data loading, drift injection, SDC and baselines (B0 single agent, B1 plain pipeline, B2 post-hoc verifier, B3 LLM judge), detection metrics, ablations, Who&When evaluation, tables and figures |
| `cache.jsonl` | Cached run records of the reported `medium` run, so the experiment can resume without re-running the LLM |

## Running the experiments

Easiest: open `notebook.ipynb` in Google Colab or Kaggle with a GPU and run all cells. The settings cell has a `PROFILE` switch:

| Profile | Model | Tasks per dataset | Drift seeds |
|---|---|---|---|
| `smoke` | Qwen2.5-1.5B-Instruct | 20 | 1 |
| `medium` (used in the paper) | Qwen2.5-3B-Instruct | 40 | 2 |
| `spec` | Qwen2.5-3B-Instruct | 100 | 1 |
| `standard` | Qwen3-8B | 50 | 3 |
| `full` | Llama-3.1-8B-Instruct, Qwen3-8B | 100 | 3 |

The reported run took about 156 minutes on two NVIDIA T4 GPUs.

**Reusing the cache.** The notebook writes results to `results_<PROFILE>/`. To resume from the cached runs, create `results_medium/` and place `cache.jsonl` inside it before running.

**Data.** HotpotQA (distractor dev set), GSM8K (test set) and the Who&When failure logs are downloaded automatically from their public sources when the notebook runs.

## Reproducibility

- Decoding is greedy. The drift seed controls which drift event each task receives. Seed 0 shuffles the tasks (12 calibration, 28 test), and seed 100 is used for calibration drift.
- Confidence intervals are percentile bootstrap (2,000 resamples) over tasks. Success differences use a paired bootstrap (5,000 resamples).
- Results can differ slightly between machines and library versions.

## Citation

```bibtex
@misc{fatima2026sdc,
  title  = {Semantic-Distance Checkpointing for Goal Drift in LLM-Based Multi-Agent Systems},
  author = {Fatima, Anabiya and Shahid, Menal},
  year   = {2026},
  note   = {Manuscript}
}
```

Update the title and details above to match the final version of the paper.
