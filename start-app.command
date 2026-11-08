#!/bin/bash
# start-app.command
# Double-clickable macOS launcher for auto-mosaic

set -e

# Change directory to the folder containing this script
cd "$(cd "$(dirname "$0")" && pwd)"

# Ensure helper scripts have executable permission
chmod +x scripts/*.sh 2>/dev/null || true

# Run setup and launch script
bash scripts/setup_and_run_mac.sh "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[auto-mosaic] Application exited with error code: $EXIT_CODE"
    read -p "Press Enter to close this window..."
fi

exit $EXIT_CODE
