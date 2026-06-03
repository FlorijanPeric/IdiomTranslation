#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKUP_DIR="$PROJECT_DIR/backups"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$BACKUP_DIR/local_pc_remote_embedder_rag_$STAMP.tar.gz"

mkdir -p "$BACKUP_DIR"

tar -czf "$ARCHIVE" \
  --exclude='local_pc_remote_embedder_rag/backups' \
  --exclude='local_pc_remote_embedder_rag/.venv311' \
  --exclude='local_pc_remote_embedder_rag/__pycache__' \
  --exclude='local_pc_remote_embedder_rag/.DS_Store' \
  -C "$(dirname "$PROJECT_DIR")" \
  "$(basename "$PROJECT_DIR")"

echo "Backup created: $ARCHIVE"
