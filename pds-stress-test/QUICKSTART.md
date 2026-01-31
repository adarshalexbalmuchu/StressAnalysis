# Quick Start Guide

## Installation

### 1. Prerequisites

- Python 3.11+
- PostgreSQL
- Poetry (or pip)

### 2. Setup

```bash
# Clone and enter directory
cd pds-stress-test

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

Or manually:

```bash
# Install dependencies
poetry install

# Create database
createdb pds_stress_test

# Setup environment
cp .env.example .env
# Edit .env with your database URL

# Run migrations
poetry run alembic upgrade head

# Start server
poetry run uvicorn app.main:app --reload
```

### 3. Verify Installation

Visit http://localhost:8000/docs - you should see the Swagger UI.

## Basic Usage

### 1. Create a Run

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "policy_rule": "Mandatory Aadhaar-based biometric authentication at FPS",
    "domain": "PDS",
    "description": "V1 stress test"
  }'
```

Save the `id` from the response.

### 2. Load Hypotheses

Use the seed data loader or API:

```bash
# Get the run ID from step 1
RUN_ID="<your-run-id>"

# Load seed data via Python
poetry run python -c "
import json
import requests
from data.seeds.loader import load_seed_file, parse_hypotheses

seed = load_seed_file('data/seeds/pds_biometric_v1.json')
hypotheses = parse_hypotheses(seed)
hypotheses_json = [h.model_dump() for h in hypotheses]

response = requests.post(
    f'http://localhost:8000/api/v1/hypotheses/bulk/${RUN_ID}',
    json=hypotheses_json
)
print(f'Loaded {len(response.json())} hypotheses')
"
```

### 3. Build Graph

```bash
curl -X POST http://localhost:8000/api/v1/graph/${RUN_ID} \
  -H "Content-Type: application/json" \
  -d @data/seeds/pds_biometric_v1.json
```

(Extract the `graph_edges` section from the seed file)

### 4. Initialize Beliefs

```bash
curl -X POST "http://localhost:8000/api/v1/beliefs/${RUN_ID}/initialize?prior_belief=0.5"
```

### 5. Run Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/simulation/${RUN_ID}?n_trajectories=1000&max_steps=10"
```

### 6. Get Results

```bash
curl http://localhost:8000/api/v1/simulation/${RUN_ID}/latest
```

## Example Workflow Script

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# 1. Create run
run = requests.post(f"{BASE_URL}/runs", json={
    "policy_rule": "Mandatory Aadhaar authentication",
    "domain": "PDS"
}).json()
run_id = run["id"]

# 2. Load hypotheses
with open("data/seeds/pds_biometric_v1.json") as f:
    seed = json.load(f)

hypotheses = requests.post(
    f"{BASE_URL}/hypotheses/bulk/{run_id}",
    json=seed["hypotheses"]
).json()

# 3. Build graph
graph = requests.post(
    f"{BASE_URL}/graph/{run_id}",
    json=seed["graph_edges"]
).json()

# 4. Initialize beliefs
beliefs = requests.post(
    f"{BASE_URL}/beliefs/{run_id}/initialize",
    params={"prior_belief": 0.5}
).json()

# 5. Run simulation
simulation = requests.post(
    f"{BASE_URL}/simulation/{run_id}",
    params={
        "n_trajectories": 1000,
        "max_steps": 10,
        "random_seed": 42
    }
).json()

# 6. Analyze results
stats = simulation["summary_statistics"]
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Avg trajectory length: {stats['avg_trajectory_length']:.1f}")
print(f"Sensitivity hotspots: {simulation['sensitivity_hotspots']}")
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app tests/

# Run specific test file
poetry run pytest tests/test_graph.py
```

## Development

### Code Style

```bash
# Format code
poetry run black app/ tests/

# Lint
poetry run ruff check app/ tests/

# Type check
poetry run mypy app/
```

### Database Migrations

```bash
# Create new migration
poetry run alembic revision --autogenerate -m "Description"

# Apply migrations
poetry run alembic upgrade head

# Rollback
poetry run alembic downgrade -1
```

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   FastAPI                       │
│            (Interface Layer Only)               │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────┐        ┌──────▼──────┐
│  Schemas   │◄───────┤   Storage   │
│ (Contract) │        │ (SQLAlchemy)│
└────────────┘        └─────────────┘
                              │
                      ┌───────▼────────┐
                      │   PostgreSQL   │
                      │   (+ Alembic)  │
                      └────────────────┘

┌─────────────────────────────────────────────────┐
│                  Engine Core                    │
│         (Pure Python - No FastAPI)              │
├─────────────────────────────────────────────────┤
│ • HypothesisGraphBuilder                       │
│ • BeliefUpdateEngine                           │
│ • MonteCarloSimulator                          │
└─────────────────────────────────────────────────┘
```

## Key Principles

1. **No LLMs for belief updates** - Pure Bayesian logic
2. **Multiple futures, not predictions** - Distribute over trajectories
3. **Explicit uncertainty** - Probabilistic belief states
4. **Engine-first** - Core logic independent of API
5. **Auditability** - Full explanation logs and reproducibility

## Troubleshooting

### Database Connection Error
```bash
# Check PostgreSQL is running
pg_isready

# Verify credentials in .env
cat .env | grep DATABASE_URL

# Test connection
psql postgresql://postgres:postgres@localhost:5432/pds_stress_test
```

### Migration Errors
```bash
# Reset migrations (CAUTION: drops all data)
poetry run alembic downgrade base
poetry run alembic upgrade head
```

### Import Errors
```bash
# Reinstall dependencies
poetry install --no-cache
```

## Next Steps

- Explore the API documentation at `/docs`
- Review the seed data in `data/seeds/pds_biometric_v1.json`
- Run tests to understand the system: `poetry run pytest -v`
- Modify simulation parameters in API calls
- Create custom hypothesis sets for different policies
