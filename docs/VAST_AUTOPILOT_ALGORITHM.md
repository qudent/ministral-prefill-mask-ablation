# Vast Training Autopilot Algorithm

## Goal
Run Stage 3 finetuning unattended with:
- early failure detection,
- bounded automatic recovery,
- low-noise monitoring,
- optional teardown on completion/failure.

## Components
- Local orchestrator: `scripts/local/vast_sleep_stage3.sh`
- Remote supervisor: `scripts/vast/remote_stage3_supervisor.sh`

## Algorithm
1. Launch remote tmux session with the supervisor.
2. Supervisor runs Stage 3 training and keeps state in:
   - `artifacts/autopilot/state.env`
   - `artifacts/autopilot/attempt_<n>.log`
3. If a run attempt fails, supervisor applies bounded auto-fixes, then retries:
   - install `uv` if missing
   - install `build-essential` if compiler is missing
   - trim old runs when disk is full
   - reduce `EVAL_SAMPLES` on CUDA OOM (when safe)
4. Local orchestrator monitors adaptively:
   - early cadence: `1, 2, 4, 8` minutes
   - stable cadence: every `30` minutes
   - alert cadence: every `3` minutes when risk signals fire
5. Alert signals:
   - traceback/runtime error markers
   - no step/log progress for consecutive checks
6. On detected stall, orchestrator can kill the stuck trainer process once, allowing supervisor retry.
7. Exit conditions:
   - `done`: successful run with summary written
   - `failed`: max attempts exhausted or unreachable
8. Optional teardown:
   - destroy Vast instance on `done` and/or `failed`

## One-Line Check Output
Each monitor cycle emits:
`ts status attempt step gpu mem fix run`

This keeps sleep-mode runs observable without high token/log noise.

## Typical Sleep Command
```bash
VAST_HOST=ssh6.vast.ai \
VAST_PORT=17956 \
INSTANCE_ID=31457957 \
MODE=prefill_bidir \
MAX_STEPS=6000 \
AUTO_STOP_PATIENCE_EVALS=8 \
AUTO_STOP_MIN_DELTA=0.001 \
AUTO_STOP_MIN_STEPS=3000 \
AUTO_DESTROY_ON_DONE=1 \
AUTO_DESTROY_ON_FAIL=1 \
bash scripts/local/vast_sleep_stage3.sh
```

