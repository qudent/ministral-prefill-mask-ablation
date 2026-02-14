#!/usr/bin/env bash
set -euo pipefail

find runs -type f -name 'log.txt' | sort | tail -n "${1:-5}" | while read -r file; do
  echo "===== $file ====="
  tail -n 40 "$file"
done
