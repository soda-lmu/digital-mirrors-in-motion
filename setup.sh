#!/bin/bash
set -e

# Install uv if not found
if ! command -v uv &> /dev/null
then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "Syncing environment with uv..."
uv sync

echo "Generating pose clusters and normalized data..."
uv run python scripts/cluster_poses.py

echo "Environment is ready!"
echo "Starting the local server..."
uv run python scripts/server.py
