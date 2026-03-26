#!/bin/bash

# ==============================================================================
# ALS-PROGNOSIS TOOL WRAPPER
# ==============================================================================
# Instructions: Edit the variables below to match your local setup.
# Run this script using: bash run_als_prognosis_docker.sh
# Supports: macOS, Linux, and Windows (via WSL2)
# ==============================================================================

# --- 1. USER SETTINGS (Edit these as needed) ---

# Where are your .csv and image files located on your computer?
INPUT_DIR="$(pwd)/data"

# The name of the CSV file inside your INPUT_DIR
INPUT_FILE="participant_inputs.csv"

# The type of input (must be t1w_maps or dbm_maps)
INPUT_TYPE="t1w_maps"

# Where should the results be saved?
OUTPUT_DIR="$(pwd)/output"

# Sub output directory, defaults to a timestamped folder
OUT_SUBDIR="results_$(date +%Y%m%d)"

# Path to the 'Pelican' data folder on your computer.
# Note: The tool will download ~3GB of data here during the first run.
# Subsequent runs will use this local copy, making them much faster.
PELICAN_DIR="$HOME/data/pelican"

# --- 2. MODEL CONFIGURATION ---

# The validated neuroimaging model for this pipeline.
# Note: Currently, only 'CALSNIC_sustain_14_reg_dbm_wscore' is supported.
MODEL_NAME="CALSNIC_sustain_14_reg_dbm_wscore"


# --- 3. PRE-EXECUTION CHECKS ---

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "❌ Error: Docker is not running. Please start Docker Desktop and try again."
  exit 1
fi

# Check if the Docker image has been built
if [[ "$(docker images -q als-prognosis-app 2> /dev/null)" == "" ]]; then
  echo "❌ Error: Docker image 'als-prognosis-app' not found."
  echo "Please run: docker build --platform linux/amd64 -t als-prognosis-app ."
  exit 1
fi

# Ensure Pelican directory exists on Mac to avoid Docker creating it as 'root'
mkdir -p "$PELICAN_DIR"

# Handle macOS "Caffeinate" (Prevents sleep during ~50-minute pelican runs)
if command -v caffeinate >/dev/null 2>&1; then
    echo "macOS detected: Using caffeinate to prevent system sleep..."
    PREFIX="caffeinate -disu"
else
    PREFIX=""
fi

# --- 4. EXECUTION ---

echo "🚀 Launching ALS-Prognosis Pipeline..."
echo "------------------------------------------------"
echo "Model:  $MODEL_NAME"
echo "Input:  $INPUT_DIR/$INPUT_FILE"
echo "Output: $OUTPUT_DIR/$OUT_SUBDIR"
echo "------------------------------------------------"

$PREFIX docker run --rm --platform linux/amd64 \
  -u $(id -u):$(id -g) \
  -v "$INPUT_DIR":/app/data \
  -v "$OUTPUT_DIR":/app/output_folder \
  -v "$PELICAN_DIR":/data/pelican \
  als-prognosis-app \
  --model "$MODEL_NAME" \
  --input_filepath "/app/data/$INPUT_FILE" \
  --input_type "$INPUT_TYPE" \
  --outdir "/app/output_folder/$OUT_SUBDIR" "$@" \
  --pelican_path "/data/pelican"


echo "------------------------------------------------"
echo "✅ Process Finished."