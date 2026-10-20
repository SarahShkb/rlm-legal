#!/bin/bash
# Runs a local DeepSeek-V4-Flash server (vLLM) on 4x H200 and then your RLM experiment
# against it, all inside one SLURM job. Lines marked CHANGE need your values.
#
# Submit with:   mkdir -p logs && sbatch run_rlm_h200.sbatch
# Watch with:    squeue -u $USER      and     tail -f logs/<jobname>_<jobid>.out

#SBATCH --job-name=rlm-dsv4
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                 # REQUIRED, without it you get no GPUs (guide 7.2)
#SBATCH --ntasks-per-node=1          # one vLLM server process that uses all 4 GPUs
#SBATCH --cpus-per-task=32           # 8 CPUs per H200 x 4 GPUs (guide Table 7.1)
#SBATCH --mem=1000G                   # CHANGE: about half the node's RAM (you use half its GPUs)
#SBATCH --time=01:00:00              # CHANGE: must be within the partition's time limit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---------- paths (CHANGE) ----------
SIF=$SCRATCH/containers/vllm.sif                 # vLLM container from step 3
MODEL_DIR=$SCRATCH/models/DeepSeek-V4-Flash      # weights from step 4
PROJECT=$HOME/thesis_s344640                             # your RLM code
VENV=$PROJECT/.venv                              # your Python env for the RLM code
EXPERIMENT_CMD="python main.py"              # CHANGE: your experiment entry point
PORT=8000

export HF_HOME=$SCRATCH/hf_cache

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi                                       # should list exactly 4 H200

# ---------- 1. start the model server in the background ----------
apptainer exec --nv --bind "$SCRATCH:$SCRATCH" "$SIF" \
  vllm serve "$MODEL_DIR" \
    --served-model-name deepseek-v4-flash \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --host 127.0.0.1 --port "$PORT" \
    > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Stop the server however the job ends, so the GPUs are released
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# ---------- 2. wait for the server (loading ~150-280 GB takes several minutes) ----------
READY=0
for i in $(seq 1 160); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "vLLM ready after $((i * 15)) s"; READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited early, see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 15
done
[ "$READY" -eq 1 ] || { echo "vLLM not ready after 40 min"; exit 1; }

# ---------- 3. run the experiment against the local server ----------
source "$VENV/bin/activate"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"   # point your client here instead of api.deepseek.com
export OPENAI_API_KEY="local"                        # vLLM ignores it, but clients want one
export MODEL_NAME="deepseek-v4-flash"                # must match --served-model-name
export LLM_PROVIDER=local            # use OPENAI_API_KEY/OPENAI_BASE_URL/MODEL_NAME above, not DeepSeek API

cd "$PROJECT"
$EXPERIMENT_CMD

echo "Experiment finished"
