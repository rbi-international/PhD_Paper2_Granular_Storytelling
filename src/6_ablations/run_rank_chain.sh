#!/usr/bin/env bash
#
# run_rank_chain.sh
#
# Runs the remaining rank sweep unattended: rank 8, then 64, then 128, back to back.
#
# Layman note: this is the overnight script. It trains each adapter one after another so
# the GPU is never sitting idle waiting for somebody to type the next command, and it
# saves and commits each rank the moment that rank finishes. If the machine dies at hour
# 12, the ranks that already completed are committed and safe rather than lost with the
# rest of the chain.
#
# WHY NO "set -e". A failure in one rank must NOT kill the chain. rank-128 is expected to
# run out of memory (measured peak at rank-32 was 5.65 GiB of about 6.0 GiB usable, and
# rank-128 needs roughly 0.9 GiB more). That OOM is a RESULT we want recorded, not an
# error that halts everything after it. The script therefore inspects each exit code and
# keeps going.
#
# Exit codes from run_rank_sweep.py:
#   0  trained and evaluated, results written, summary row appended
#   2  out of memory, recorded in its own experiment folder, summary deliberately untouched
#   *  anything else is a real failure, logged and skipped, chain continues
#
# Resume: safe to re-run. A rank whose adapter is already saved is reused rather than
# retrained, and a rank interrupted mid-training resumes from its last checkpoint. The
# evaluation half resumes per row on the frozen prompt_id.
#
# Checkpoints are deliberately NOT auto-deleted. They are about 3.5 GB per rank and disk
# has room; an unattended script should not be deleting things while nobody is watching.
# Clean them up by hand afterwards if the space is wanted:
#     rm -rf models/rank_sweep/checkpoints
#
# Usage:
#     bash src/6_ablations/run_rank_chain.sh
#
# No em dashes anywhere (project style rule).

set -u

PROJECT_ROOT="e:/PHDETP2/PhD_Paper2_Granular_Storytelling/PhD_Paper2_Granular_Storytelling"
PY="C:/Users/rbhar/anaconda3/envs/study_torch/python.exe"
LOG_DIR="$PROJECT_ROOT/logs/rank_chain"

export KMP_DUPLICATE_LIB_OK=TRUE

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$LOG_DIR"

CHAIN_LOG="$LOG_DIR/chain.log"

say() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"
}

say "=== rank chain starting: 8, then 64, then 128 ==="
say "commit at launch: $(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    say "WARNING: tree is not clean at launch, recorded commit hashes will be incomplete"
fi

for RANK in 8 64 128; do
    say "--- rank $RANK starting ---"
    RANK_LOG="$LOG_DIR/rank_${RANK}.log"

    "$PY" src/6_ablations/run_rank_sweep.py --rank "$RANK" >> "$RANK_LOG" 2>&1
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        ACC=$(grep -o "Top-1 accuracy [0-9.]*%" "$RANK_LOG" | tail -1)
        say "rank $RANK COMPLETE, $ACC"

        # Commit this rank on its own, so a later crash cannot take it down with it.
        git add "rank_${RANK}_results.csv" results/all_configs_summary.csv \
                "experiments/"*"_rank_${RANK}" 2>/dev/null
        git commit -q -m "Rank sweep: r=${RANK} complete, ${ACC}

One point of the clean LoRA-rank sweep. Trained with everything except r frozen from the
production recipe (alpha stays 64, so effective scaling alpha/rank moves with rank), then
evaluated through the Task A path on the revision manifest with steering on.

Interpretation is bounded by two things fixed before any numbers existed: the retraining
noise floor of 3.13 points measured from rank_32 against decoding_topp, and the narrow
capacity claim (lm_head and wte train in full at every rank, so 16x rank is only 1.66x
trainable capacity, which supports a claim about LoRA rank specifically and not about
adapter capacity in general)." \
            && say "rank $RANK committed" \
            || say "rank $RANK commit reported nothing to do"

    elif [ $STATUS -eq 2 ]; then
        say "rank $RANK OUT OF MEMORY (expected at 128). Recorded, summary untouched, chain continues."
        git add "experiments/"*"_rank_${RANK}_oom" 2>/dev/null
        git commit -q -m "Rank sweep: r=${RANK} exceeded the 6GB VRAM budget

Recorded as a measured hardware limit rather than a missing data point, which is on-thesis
for a resource-constrained paper. The adapter never trained, so no results CSV exists and
no summary row was written. Provenance for the attempt is in its experiment folder.

Predicted in advance from the rank-32 measurement: peak there was 5.65 GiB of about 6.0
GiB usable, and this rank needs roughly 0.9 GiB more for optimizer state." \
            && say "rank $RANK OOM record committed" \
            || say "rank $RANK OOM commit reported nothing to do"

    else
        say "rank $RANK FAILED with exit code $STATUS. See $RANK_LOG. Chain continues."
    fi
done

say "=== rank chain finished ==="
say "summary now holds:"
grep -c "" results/all_configs_summary.csv | xargs -I{} say "  {} lines (including header)"
grep "^rank_" results/all_configs_summary.csv | cut -d, -f1,5 | while read -r line; do
    say "  $line"
done
say "Checkpoints kept at models/rank_sweep/checkpoints (about 3.5 GB per rank)."
say "Next: figures, the new paper-table generator, then push."
