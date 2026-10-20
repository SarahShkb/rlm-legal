#!/bin/bash
# Runs a local Qwen3-30B-A3B-Instruct-2507 server (vLLM) on 2x A40 and then your RLM
# experiment against it, in one SLURM job. Lines marked CHANGE need your values.
#
# Submit:  mkdir -p logs && sbatch run_rlm_a40.sbatch
# Watch:   squeue -u $USER     and     tail -f logs/<jobname>_<jobid>.out

#SBATCH --job-name=rlm-qwen
#SBATCH --partition=gpu_a40          # open partition, 24 h max (gpu_a40_ext: 120 h)
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                 # REQUIRED, without it you get no GPUs
#SBATCH --ntasks-per-node=1          # one vLLM server process using both GPUs
#SBATCH --cpus-per-task=24           # 12 CPUs per A40 x 2 (guide Table 7.1)
#SBATCH --mem=192G                   # guide's example for 2x A40
#SBATCH --time=01:00:00              # CHANGE: 1 h for the test, longer for full runs (max 24 h)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---------- paths (CHANGE if needed) ----------
SIF=$SCRATCH/containers/vllm.sif
MODEL_DIR=$SCRATCH/models/Qwen3-30B-A3B-Instruct-2507     # from download_qwen.py
PROJECT=$HOME/thesis_s344640                               # your git clone
VENV=$PROJECT/.venv
EXPERIMENT_CMD="python -u main.py"                     # CHANGE: your entry point (-u = live log)
MAX_LEN=262144      # model's native max; set lower (e.g. 32768) for capped-context runs
PORT=8000

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1             # never try to download inside the job
export PYTHONUNBUFFERED=1           # print output to the log immediately

# Random key for this job only: the server requires it, your code reads it
export OPENAI_API_KEY=$(openssl rand -hex 16)

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi                           # should list exactly 2 A40

# ---------- 1. start the model server in the background ----------
apptainer exec --nv --bind "$HOME:$HOME" "$SIF" \
  vllm serve "$MODEL_DIR" \
    --served-model-name qwen3-30b-a3b \
    --api-key "$OPENAI_API_KEY" \
    --tensor-parallel-size 2 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 --port "$PORT" \
    > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT   # always free the GPUs at the end

# ---------- 2. wait until the server is ready ----------
READY=0
for i in $(seq 1 80); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "vLLM ready after $((i * 15)) s"; READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited early, see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 15
done
[ "$READY" -eq 1 ] || { echo "vLLM not ready after 20 min"; exit 1; }

# ---------- 3. run the experiment against the local server ----------
source "$VENV/bin/activate"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export MODEL_NAME="qwen3-30b-a3b"    # must match --served-model-name
export LLM_PROVIDER=local            # use OPENAI_API_KEY/OPENAI_BASE_URL/MODEL_NAME above, not DeepSeek

cd "$PROJECT"
$EXPERIMENT_CMD
echo "Experiment finished"
