#!/bin/bash

# ====== Configurable Parameters ======
MODEL="2x.model"
WEIGHTS="rl-from-early-game-2x.weights"
TARGET_ITEM="iron_ingot"
MAX_STEPS=2500
EPISODES=101  # Number of episodes to run     
SAVE_DIR="Data/$TARGET_ITEM"

# ====== Run Command ======
xvfb-run python run_agent.py \
  --model "$MODEL" \
  --weights "$WEIGHTS" \
  --target-item "$TARGET_ITEM" \
  --max-steps "$MAX_STEPS" \
  --episodes "$EPISODES" \
  --save-dir "$SAVE_DIR"
