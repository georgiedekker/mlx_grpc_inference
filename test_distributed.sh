#!/bin/bash
# Test distributed inference setup

echo "=== Testing Distributed Inference ==="
echo "1. Starting coordinator on mini1..."
./launch_coordinator.sh &
COORD_PID=$!

echo "2. Waiting for coordinator to initialize..."
sleep 10

echo "3. Checking health..."
curl http://localhost:8100/health | jq

echo "4. Testing inference..."
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 20,
    "temperature": 0.7
  }' | jq

echo "5. Cleaning up..."
kill $COORD_PID

echo "=== Test complete ==="