#!/bin/bash

# ====== Configurable Parameters ======
MODEL="2x.model"
WEIGHTS="rl-from-early-game-2x.weights"
TARGET_ITEM="wooden_pickaxe"
MAX_STEPS=1000
EPISODES=3  # Number of episodes to run     
SAVE_DIR="Data/$TARGET_ITEM"

# ====== Run Command ======
xvfb-run python run_agent.py \
  --model "$MODEL" \
  --weights "$WEIGHTS" \
  --target-item "$TARGET_ITEM" \
  --max-steps "$MAX_STEPS" \
  --episodes "$EPISODES" \
  --save-dir "$SAVE_DIR"

# ====== Run the get_data_stats file ======
python get_data_stats.py \
  --data-dir "$SAVE_DIR"

# ====== Print that we are done ======
echo "Data generation and statistics collection completed."