#!/usr/bin/env zsh
set -euo pipefail

# Start Colima (safe if already running)
if ! colima status >/dev/null 2>&1; then
  echo "Starting Colima..."
  colima start --cpu 4 --memory 6 --disk 40
else
  echo "Colima is already running."
fi

# Ensure Docker CLI uses Colima context
if docker context ls | grep -q 'colima'; then
  docker context use colima >/dev/null
fi

# Start local Qdrant container from compose
if command -v docker-compose >/dev/null 2>&1; then
  docker-compose up -d
else
  docker compose up -d qdrant
fi

echo "Qdrant should be available at: http://127.0.0.1:6333"
