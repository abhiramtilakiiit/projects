#!/usr/bin/env bash
# ==============================================================================
# Psychological Experiment Multi-Port Server Runner
# Launches static HTTP servers for experimental conditions (ports 8001-8006)
# and a landing router page (port 8000).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-0.0.0.0}"
PIDS=()

cleanup() {
    echo ""
    echo "[!] Shutting down all experimental HTTP servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "[✓] All servers terminated."
}

trap cleanup EXIT INT TERM

echo "======================================================================"
echo " Starting MMT Experiment Condition Servers on ${HOST}"
echo "======================================================================"

# 1. Start Landing Router (Port 8000)
cd "${SCRIPT_DIR}"
python3 -m http.server 8000 --bind "${HOST}" >/dev/null 2>&1 &
PIDS+=($!)
echo " [Landing Router] -> http://${HOST}:8000"

# 2. Start Condition Servers (Ports 8001-8006)
CONDITIONS=(
    "8001:Group 1 (Sad Music + Happy Narrative - Incongruent)"
    "8002:Group 2 (Sad Music + Sad Narrative - Congruent)"
    "8003:Group 3 (Sad Music + No Narrative - Control)"
    "8004:Group 4 (Happy Music + Happy Narrative - Congruent)"
    "8005:Group 5 (Happy Music + Sad Narrative - Incongruent)"
    "8006:Group 6 (Happy Music + No Narrative - Control)"
)

for entry in "${CONDITIONS[@]}"; do
    PORT="${entry%%:*}"
    LABEL="${entry#*:}"
    DIR_NUM="$((PORT - 8000))"

    if [[ -d "${SCRIPT_DIR}/${DIR_NUM}" ]]; then
        (
            cd "${SCRIPT_DIR}/${DIR_NUM}"
            python3 -m http.server "${PORT}" --bind "${HOST}" >/dev/null 2>&1
        ) &
        PIDS+=($!)
        echo " [${PORT}] ${LABEL} -> http://${HOST}:${PORT}"
    else
        echo " [!] Warning: Condition directory ${DIR_NUM} not found."
    fi
done

echo "======================================================================"
echo " Experiment servers are live. Press Ctrl+C to terminate."
echo "======================================================================"

# Wait for all processes
wait
