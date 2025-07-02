# #!/bin/bash
# set -e

# PY_SCRIPT=$(find ./src -maxdepth 1 -type f -name "*.py" | head -n 1)
# NB_SCRIPT=$(find ./src -maxdepth 1 -type f -name "*.ipynb" | head -n 1)

# if [ -n "$PY_SCRIPT" ]; then
#   echo "▶️ Running Python script: $PY_SCRIPT"
#   exec python3 -u "$PY_SCRIPT"

# elif [ -n "$NB_SCRIPT" ]; then
#   echo "📓 Found Notebook: $NB_SCRIPT"

#   echo "✅ Registering current environment as user-kernel..."
#   python -m ipykernel install --user --name user-kernel --display-name "User Kernel"

#   echo "⚡ Patching kernelspec to use user-kernel..."
#   jq '.metadata.kernelspec.name="user-kernel" | .metadata.kernelspec.display_name="User Kernel"' "$NB_SCRIPT" > tmp && mv tmp "$NB_SCRIPT"

#   echo "⚡ Executing notebook with user-kernel..."
#   exec jupyter nbconvert --to notebook --execute --inplace "$NB_SCRIPT" --ExecutePreprocessor.kernel_name="user-kernel"

# else
#   echo "❌ No .py or .ipynb file found in ./src/"
#   exit 1
# fi


# !/bin/bash
set -e

echo "✅ Registering current environment as user-kernel..."
python -m ipykernel install --user --name user-kernel --display-name "User Kernel"

PY_SCRIPT=$(find ./src -maxdepth 1 -type f -name "*.py" | head -n 1)
NB_SCRIPT=$(find ./src -maxdepth 1 -type f -name "*.ipynb" | head -n 1)

if [ -n "$PY_SCRIPT" ]; then
  echo "▶️ Running Python script: $PY_SCRIPT"
  exec python3 -u "$PY_SCRIPT"

elif [ -n "$NB_SCRIPT" ]; then
  echo "📓 Found Notebook: $NB_SCRIPT"

  echo "⚡ Patching kernelspec to use user-kernel..."
  jq '.metadata.kernelspec.name="user-kernel" | .metadata.kernelspec.display_name="User Kernel"' "$NB_SCRIPT" > tmp && mv tmp "$NB_SCRIPT"

  echo "⚡ Switching to notebook working directory..."
  cd "$(dirname "$NB_SCRIPT")"

  echo "⚡ Executing notebook with user-kernel..."
  exec jupyter nbconvert --to notebook --execute --inplace "$(basename "$NB_SCRIPT")" --ExecutePreprocessor.kernel_name="user-kernel"

else
  echo "❌ No .py or .ipynb file found in ./src/"
  exit 1
fi
