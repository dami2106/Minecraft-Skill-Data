#!/bin/bash

# ====== Task to Run ======
TARGET_ITEM="iron_ingot"  # Change this to run a different task

# ====== Configurable Parameters ======
MODEL="2x.model"
WEIGHTS="rl-from-early-game-2x.weights"
EPISODES=100
SAVE_DIR="Data/$TARGET_ITEM"

# ====== Step Count Lookup ======
declare -A STEP_LOOKUP=(
  ["wooden_pickaxe"]=800
  ["cobblestone"]=1000
  ["stone_pickaxe"]=1500
  ["iron_ingot"]=2500
)

# ====== Check for valid task ======
if [[ -z "${STEP_LOOKUP[$TARGET_ITEM]}" ]]; then
  echo "Error: Unknown target item '$TARGET_ITEM'."
  echo "Valid options are: ${!STEP_LOOKUP[@]}"
  exit 1
fi

MAX_STEPS="${STEP_LOOKUP[$TARGET_ITEM]}"

# ====== Run Command ======
echo "Running task: $TARGET_ITEM with $MAX_STEPS steps."

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

# ====== Done ======
echo "Data generation and statistics collection completed for $TARGET_ITEM."
