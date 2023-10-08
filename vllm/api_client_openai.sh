#!/bin/bash

set -e

HOST="localhost"
PORT="8000"
URL="http://""$HOST"":""$PORT"
ENDPOINT_MODELS_LIST="$URL""/v1/models"
ENDPOINT_QUERY="$URL""/v1/completions"

echo "Listing available models."
curl "$ENDPOINT_MODELS_LIST"

echo "Test query..."
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "facebook/opt-125m",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'