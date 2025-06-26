#!/bin/bash
set -e

PY_SCRIPT=$(find ./src -maxdepth 1 -type f -name "*.py" | head -n 1)
NB_SCRIPT=$(find ./src -maxdepth 1 -type f -name "*.ipynb" | head -n 1)

if [ -n "$PY_SCRIPT" ]; then
  echo "▶️ Running Python script: $PY_SCRIPT"
  exec python3 -u "$PY_SCRIPT"
elif [ -n "$NB_SCRIPT" ]; then
  echo "📓 Running Notebook: $NB_SCRIPT"
  exec jupyter nbconvert --to notebook --execute --inplace "$NB_SCRIPT"
else
  echo "❌ No .py or .ipynb file found in ./src/"
  exit 1
fi
