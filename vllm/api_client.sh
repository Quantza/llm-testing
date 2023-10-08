#!/bin/bash

set -e

HOST="localhost"
PORT="8000"
URL="http://""$HOST"":""$PORT"
ENDPOINT="$URL""/generate"

conda activate "$DEV_VENV"

echo "Test query..."
curl "$ENDPOINT" \
-d '{
"prompt": "San Francisco is a",
"use_beam_search": true,
"n": 4,
"temperature": 0
}'