#!/bin/bash
RUN_ID="6e8a898e-425a-4d82-a7d8-2180fd771be7"

echo "=== STEP 2: Upload Hypotheses ==="
curl -s -X POST "http://localhost:8000/api/v1/runs/$RUN_ID/hypotheses" \
  -H "Content-Type: application/json" \
  -d '{"hypotheses": [{"name": "H1", "description": "Test"}]}' | jq -r '.hypotheses_count'

echo "=== STEP 3: Validate Graph ==="
curl -s -X POST "http://localhost:8000/api/v1/runs/$RUN_ID/graph/validate-and-save" \
  -H "Content-Type: application/json" \
  -d '{"edges": [{"source": "H001", "target": "H001", "label": "self"}]}' | jq -r '.is_valid'

echo "=== STEP 4: Upload Signals ==="
curl -s -X POST "http://localhost:8000/api/v1/runs/$RUN_ID/signals" \
  -H "Content-Type: application/json" \
  -d '{"signals": [{"name": "S1", "description": "Test", "source": "test", "timestamp": "2026-01-31T00:00:00Z", "content": "test"}]}' 

echo -e "\n=== STEP 5: Initialize Priors ==="
curl -s -X POST "http://localhost:8000/api/v1/runs/$RUN_ID/belief/init"

echo -e "\n=== STEP 6: Update Beliefs ==="
curl -s -X POST "http://localhost:8000/api/v1/runs/$RUN_ID/belief/update"

echo -e "\n=== STEP 7: Simulate ==="
curl -s -X POST "http://localhost:8000/api/v1/runs/$RUN_ID/simulate" \
  -H "Content-Type: application/json" \
  -d '{"params": {"n_runs": 100}}'
