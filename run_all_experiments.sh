#!/bin/bash

# This script runs all the training variations to generate data for analysis.
# WARNING: This will take a very long time and consume significant computational resources.

# --- Configuration ---
# Define the seeds you want to run for each experiment
SEEDS=(123 456 789) # Using 3 seeds as an example. For robust results, 5-10 seeds are recommended.

# Set the config to use ('test' is for quick checks, 'default' for full runs)
CONFIG="default"

# --- Pre-requisites for Dream and Dyna Agents ---
# The 'dream' agents require a pre-trained world model and a replay buffer.
# train a world model using `train_transformer_world_model.py` or `train_world_model_parallel.py` (for GRU)
# and generate a buffer by running the PPO agent for a while and saving the buffer.

# NOTE: Replace these with actual paths to your files
WM_TRANSFORMER_CHECKPOINT="checkpoints/transformer_wm_checkpoints/default_transformer_wm_step_200000.pth"
WM_GRU_CHECKPOINT="checkpoints/gru_wm_checkpoints/default_gru_wm_step_200000.pth"
REPLAY_BUFFER="data/replay_buffer_200k.pt" # Example path

# Check if prerequisite files exist
if [ ! -f "$WM_TRANSFORMER_CHECKPOINT" ]; then
    echo "Warning: Transformer WM checkpoint not found at $WM_TRANSFORMER_CHECKPOINT"
fi
if [ ! -f "$WM_GRU_CHECKPOINT" ]; then
    echo "Warning: GRU WM checkpoint not found at $WM_GRU_CHECKPOINT"
fi
if [ ! -f "$REPLAY_BUFFER" ]; then
    echo "Warning: Replay buffer not found at $REPLAY_BUFFER. Dream agents will fail."
fi


# --- Training Runs ---

# 1. Model-Free PPO Baseline (ppo_sb3)
# -------------------------------------
echo "--- Starting Model-Free PPO (ppo_sb3) runs ---"
for seed in "${SEEDS[@]}"; do
  RUN_NAME="ppo_sb3_seed${seed}"
  echo "Running ${RUN_NAME}..."
  python -m src.train_ppo_sb3 --config "$CONFIG" --seed "$seed" --run-name "$RUN_NAME"
done

# 2. Dream Training (GRU)
# -------------------------
echo "\n--- Starting Dream GRU runs ---"
for seed in "${SEEDS[@]}"; do
  RUN_NAME="ppo_dream_gru_seed${seed}"
  echo "Running ${RUN_NAME}..."
  python -m src.train_ppo_in_dream \
    --config "$CONFIG" \
    --wm-type gru \
    --wm-checkpoint "$WM_GRU_CHECKPOINT" \
    --buffer "$REPLAY_BUFFER" \
    --seed "$seed" \
    --run-name "$RUN_NAME"
done

# 3. Dream Training (Transformer)
# -------------------------------
echo "\n--- Starting Dream Transformer runs ---"
for seed in "${SEEDS[@]}"; do
  RUN_NAME="ppo_dream_transformer_seed${seed}"
  echo "Running ${RUN_NAME}..."
  python -m src.train_ppo_in_dream \
    --config "$CONFIG" \
    --wm-type transformer \
    --wm-checkpoint "$WM_TRANSFORMER_CHECKPOINT" \
    --buffer "$REPLAY_BUFFER" \
    --seed "$seed" \
    --run-name "$RUN_NAME"
done

# 4. Dyna-Style Training (GRU)
# ----------------------------
echo "\n--- Starting Dyna GRU runs ---"
for seed in "${SEEDS[@]}"; do
  RUN_NAME="dyn_gru_seed${seed}"
  echo "Running ${RUN_NAME}..."
  python -m src.train_dyna_loop \
    --config "$CONFIG" \
    --world-model-type gru \
    --seed "$seed" \
    --run-name "$RUN_NAME"
done

# 5. Dyna-Style Training (Transformer)
# ------------------------------------
echo "\n--- Starting Dyna Transformer runs ---"
for seed in "${SEEDS[@]}"; do
  RUN_NAME="dyn_transformer_seed${seed}"
  echo "Running ${RUN_NAME}..."
  python -m src.train_dyna_loop \
    --config "$CONFIG" \
    --world-model-type transformer \
    --seed "$seed" \
    --run-name "$RUN_NAME"
done


echo "\n--- All training runs launched! ---"
echo "Once completed, you can analyze the results by running:"
echo "python src/analyze_results.py"
