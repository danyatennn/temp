#!/usr/bin/env bash
#
# Throwaway helper: train every trainable model one after another (overnight).
# Delete this file when done; do NOT commit it.
#
# Launch (and keep the Mac awake the whole time):
#     export COMET_API_KEY=your_key            # optional, enables online logging
#     caffeinate -is bash run_overnight.sh
#
# Tunables via env vars:
#     EPOCHS=18         epochs per model
#     TRAIN_LIMIT=0     training images per epoch (0 = full dataset)
#     CACHE=data/hf_cache   shared dataset + PSF cache (built once, reused)
# e.g. (quick subset on a laptop):
#     EPOCHS=5 TRAIN_LIMIT=2000 caffeinate -is bash run_overnight.sh
#
# Per-model logs: train_logs/<name>.log    master log: train_logs/run.log

cd "$(dirname "$0")" || exit 1

PY=./venv/bin/python
export PYTHONPATH=.

EPOCHS=${EPOCHS:-18}
TRAIN_LIMIT=${TRAIN_LIMIT:-0}
CACHE=${CACHE:-data/hf_cache}

LOGDIR=train_logs
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/run.log"

# overrides shared by every run
COMMON=( "+datasets.train.cache_dir=$CACHE" "+datasets.test.cache_dir=$CACHE"
         "trainer.n_epochs=$EPOCHS" )
if [ "$TRAIN_LIMIT" -gt 0 ]; then
  COMMON+=( "+datasets.train.limit=$TRAIN_LIMIT" )
fi
if [ -z "$COMET_API_KEY" ]; then
  echo "WARNING: COMET_API_KEY not set -> using offline Comet logging." | tee -a "$MASTER"
  COMMON+=( "writer.mode=offline" )
fi

stamp () { date '+%F %T'; }

run () {
  NAME="$1"; shift
  echo "=== [$(stamp)] START $NAME ===" | tee -a "$MASTER"
  $PY train.py "$@" writer.run_name="$NAME" < /dev/null > "$LOGDIR/$NAME.log" 2>&1
  echo "=== [$(stamp)] END   $NAME (exit $?) ===" | tee -a "$MASTER"
}

# --- smoke test: fail fast (also does the one-time dataset + PSF caching) ---
echo "=== [$(stamp)] SMOKE TEST (downloads data + builds PSF cache first time) ===" \
  | tee -a "$MASTER"
$PY train.py model=unrolled_admm20 \
    "+datasets.train.cache_dir=$CACHE" "+datasets.test.cache_dir=$CACHE" \
    "+datasets.train.limit=8" "datasets.test.limit=8" \
    trainer.n_epochs=1 dataloader.batch_size=2 \
    writer.mode=offline writer.run_name=smoke_test trainer.override=True \
    < /dev/null > "$LOGDIR/smoke_test.log" 2>&1
if [ $? -ne 0 ]; then
  echo "SMOKE TEST FAILED -> see $LOGDIR/smoke_test.log. Aborting." | tee -a "$MASTER"
  exit 1
fi
rm -rf saved/smoke_test
echo "=== [$(stamp)] smoke test passed, starting full training ===" | tee -a "$MASTER"

# --- required models (modular_pre_post already trained, skipped) ---
run unrolled_admm20  model=unrolled_admm20  "${COMMON[@]}"
run modular_pre      model=modular_pre      "${COMMON[@]}"
run modular_post     model=modular_post     "${COMMON[@]}"

# --- bonus models (trainable ones) ---
run fista_unrolled   model=fista_unrolled   "${COMMON[@]}"
# Real-ESRGAN fine-tuning is the slowest (runs ADMM-100 every step); kept last.
run realesrgan_ft    model=admm100_realesrgan_ft "${COMMON[@]}" \
    optimizer.lr=1e-5 dataloader.batch_size=2

echo "=== [$(stamp)] ALL DONE ===" | tee -a "$MASTER"
