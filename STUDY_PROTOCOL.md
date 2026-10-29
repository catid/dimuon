# Prospective equal-effort optimizer study

`run_study.py` preregisters and runs a new supervised fine-tuning study. It does
not reinterpret the short historical pilot as confirmatory evidence. It plans
by default; `--execute` is required to launch training. This protocol concerns
teacher-forced cross-entropy, not RL reward or generated-answer accuracy.

## Fixed budgets and selection

The default design covers GSM8K and MATH, conventional LoRA, full NoRA
(normalization on every forward), and IniLoRA-alpha. Every optimizer in every
dataset/parameterization cell receives the following identical budget:

| Item | Default for AdamW, Muon, and Dimuon |
| --- | --- |
| LR candidates | 3, declared before training |
| Tuning seeds per LR | 1 (`1234`) |
| Updates per run | 200, including 10 timing warmups |
| Validation batches per tuning run | 8 global batches |
| Confirmation seeds for selected LR | 3 (`2345,3456,4567`), disjoint from tuning |
| Test batches per confirmation run | 32 global batches |
| Global training batch | 8 GPUs × 2 examples × 512 packed tokens |
| Adapter shape | Rank 32, alpha 32, FFN projections in the last 64 layers |

This is 54 tuning runs and 54 confirmation runs, or 21,600 optimizer updates
in total. Budgets are equal in update count and token count, not wall-clock
time. Throughput and optimizer-state memory remain separate outcomes. Validation
and test budgets can differ from each other; each budget is identical across
optimizers and parameterizations. Warmup updates count toward the 200 updates.

The predeclared LR grids are:

| Parameterization | AdamW | Muon | Dimuon |
| --- | --- | --- | --- |
| LoRA | 0.00003, 0.0001, 0.0003 | 0.0003, 0.001, 0.003 | 0.003, 0.012, 0.048 |
| NoRA | 0.00003, 0.0001, 0.0003 | 0.0003, 0.001, 0.003 | 0.001, 0.003, 0.012 |
| IniLoRA-alpha | 0.00003, 0.0001, 0.0003 | 0.0003, 0.001, 0.003 | 0.001, 0.003, 0.012 |

These grids incorporate prior pilot information. Equal candidate counts do not
establish equal historical tuning effort or globally optimal LRs. A custom JSON
grid is accepted, but every optimizer/parameterization must have the same number
of distinct positive candidates. The manifest and plan preserve the actual grid.
Datasets, parameterizations, seeds, update counts, and batch budgets are configurable
before the first run; a restricted study should be reported as such.

All tuning jobs finish before any test evaluation. For each dataset,
parameterization, and optimizer, select the LR with the lowest mean validation CE
over **all** tuning seeds. Exact ties choose the smaller LR. A candidate with a
failed seed is ineligible; the failure and any partial output remain in the
artifacts. If every candidate fails, that cell has no winner and no confirmation
run. No additional candidate is substituted.

Freeze selections in `selection.json`, retrain each selected LR from the original
checkpoint at the three fresh confirmation seeds, and evaluate only on the test
manifest. The existing harness accepts one `--eval-manifest`, so tuning passes the
validation manifest and confirmation passes the test manifest. Confirmation does
not use test loss to choose a seed, LR, stopping point, parameterization, or
optimizer. Report all seeds, failed cells, test means and sample standard
deviations. With only three seeds these are descriptive statistics, not a strong
significance claim. If confirmation fails for some seeds, the summary marks the
cell incomplete; its mean over available seeds is not a complete result.

## Data and model requirements

Supply portable local paths to one checkpoint and six JSONL manifests. Each
record must have string `question` and `gold` fields. The launcher hashes the
complete manifest files and rejects question overlaps between differently named
splits, including across datasets, after case/whitespace normalization. This is
an exact-text leakage check, not semantic deduplication. Split files must already
be prepared independently of the results. The launcher reads test text for input
validation and overlap checking only; no test metrics enter LR selection.

The current harness packs question/prompt text and short answers, trains on all
tokens, and evaluates the first fixed packed batches. CE includes prompt text;
it is not response-only CE or reasoning-task accuracy. `--max-rows` caps training
records, and the harness independently caps evaluation records to
`max(batch × eval_steps × 8, 64)`. Evaluation budgets therefore describe packed
tokens/batches, not a full benchmark split. Training sequences and evaluation
sequences are fixed for all arms. Seeds change adapter initialization, not the
data order. The sequence construction may wrap a short corpus; prepare enough
data to avoid unwanted repeated examples. These choices should be disclosed.

The runner accepts the current harness's parameterizations as implemented.
IniLoRA-alpha's BF16 residual subtraction changes the effective initial model;
this study compares optimizers **within** each parameterization and does not
claim equal starting logits across parameterizations. Inspect/fix the
parameterization separately before making a paper-faithful IniLoRA claim.

## Invocation

Use the Python executable from the training environment. The launcher itself has
no dependencies beyond Python's standard library and runs on Linux (file locks).

```bash
python3 run_study.py \
  --python /path/to/venv/bin/python \
  --model /path/to/Qwen2.5-32B-Instruct \
  --gsm8k-train /path/to/gsm8k_train.jsonl \
  --gsm8k-val /path/to/gsm8k_val.jsonl \
  --gsm8k-test /path/to/gsm8k_test.jsonl \
  --math-train /path/to/math_train.jsonl \
  --math-val /path/to/math_val.jsonl \
  --math-test /path/to/math_test.jsonl \
  --out-dir /path/to/study-results
```

This writes `plan.json` and launches no jobs. Inspect the plan, then repeat the
same command with `--execute`. Use `--phase tune` to stop after tuning and frozen
selection; repeat with `--phase confirm --execute` for confirmation. The phase is
an execution control, not a protocol change. `--gpus` defaults to
`0,1,2,3,4,5,6,7` and must name exactly eight distinct devices. Each job invokes
the specified Python's `torch.distributed.run` with eight processes. Jobs run
sequentially; optimizer order rotates deterministically across cells/candidates/
seeds. Keep other GPU workloads stopped when measuring speed.

The runner uses a nonblocking shared file lock, default
`~/.cache/dimuon/study.lock`, held for the entire invocation. This prevents two
cooperating study runners under the same user from sharing the GPUs, even with
different output directories. It cannot stop unrelated GPU programs or another
user; if sharing hardware, coordinate one `--lock-file` path with all runners.

## Resumption and artifacts

`plan.json` includes full configuration, seed/LR grids, manifest SHA-256 hashes,
runner/harness/optimizer source SHA-256 hashes, installed package versions, and
checkpoint identity. Small checkpoint files are hashed; large weight files are
identified by path, size, and modification time, **not full content hashes**.
Keep the checkpoint immutable and record its upstream revision independently.
Software versions and the plan are checked again on resume; code, manifests,
and model identity are checked before each new job.

Reusing an output directory with a changed protocol, code, environment, or input
identity fails. Start a new output directory for a new study. A matching filename
alone never establishes completion. A successful cached job requires a matching
plan/job receipt, the recorded result-file SHA-256, matching output arguments
(including seed, initialization, LR and evaluation manifest), exactly one result,
eight ranks, the full expected loss history, no nonfinite steps, and finite
positive quality/timing/state fields. Modified or incomplete successful results
are not silently reused.

Each attempt is kept separately under `jobs/.../attempt_NNNN/`, with a receipt,
exact command, combined stdout/stderr, and raw JSON output when available. A
failed job is a terminal outcome for that predeclared candidate/seed; resuming
does not secretly retry numerical failures. An interrupted process has status
`interrupted` and can be retried in a new attempt. The command stops its entire
launcher process group on Ctrl-C or timeout. Existing attempts are never erased.
Once selected, LRs cannot change within that study: changes to the validated
tuning evidence make confirmation stop rather than replace the frozen selection.

`summary.json` preserves selected LRs, all confirmation seed results/statuses,
means and sample standard deviations, and completion counts. Retain failures
when publishing aggregate results. The harness saves no trained checkpoint;
retraining for confirmation is intentional. Use an output directory outside the
repository, and publish only reviewed small manifests/results/figures. Do not
commit model weights, datasets, checkpoints, or large logs.

The job schedule, seeds, inputs and selection rule are deterministic. Bitwise GPU
reproducibility is not guaranteed: the current harness does not enforce
`torch.use_deterministic_algorithms`, and CUDA/DDP kernels can be nondeterministic.
`PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` are fixed per job, but they do not
remove that limitation.
