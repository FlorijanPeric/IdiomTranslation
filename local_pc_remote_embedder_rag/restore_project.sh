#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKUP_DIR="$PROJECT_DIR/backups"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./restore_project.sh <backup-tar.gz>"
  echo "Example: ./restore_project.sh backups/local_pc_remote_embedder_rag_20260505_120000.tar.gz"
  exit 1
fi

ARCHIVE="$1"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "Backup file not found: $ARCHIVE"
  exit 1
fi

echo "This will replace project files in: $PROJECT_DIR"
read "REPLY?Type 'YES' to continue: "
if [[ "$REPLY" != "YES" ]]; then
  echo "Restore cancelled."
  exit 1
fi

rm -rf "$PROJECT_DIR"
mkdir -p "$(dirname "$PROJECT_DIR")"
tar -xzf "$ARCHIVE" -C "$(dirname "$PROJECT_DIR")"

echo "Restore complete from: $ARCHIVE"
