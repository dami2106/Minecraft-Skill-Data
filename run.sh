#!/bin/bash

# ====== Configurable Parameters ======
MODEL="2x.model"
WEIGHTS="rl-from-early-game-2x.weights"
TARGET_ITEM="wooden_pickaxe"
MAX_STEPS=700
EPISODES=10
SAVE_DIR="Data/$TARGET_ITEM"

# ====== Run Command ======
xvfb-run python run_agent.py \
  --model "$MODEL" \
  --weights "$WEIGHTS" \
  --target-item "$TARGET_ITEM" \
  --max-steps "$MAX_STEPS" \
  --episodes "$EPISODES" \
  --save-dir "$SAVE_DIR"
