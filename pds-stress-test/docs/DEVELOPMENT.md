# Development Guide

## Project Structure

```
pds-stress-test/
├── app/
│   ├── api/v1/              # FastAPI routes (interface only)
│   │   ├── runs.py          # Run management endpoints
│   │   ├── hypotheses.py    # Hypothesis CRUD endpoints
│   │   ├── graph.py         # Graph construction endpoints
│   │   ├── beliefs.py       # Belief state endpoints
│   │   ├── signals.py       # Evidence signal endpoints
│   │   └── simulation.py    # Simulation execution endpoints
│   ├── engine/              # Core logic (NO FastAPI imports)
│   │   ├── graph.py         # Hypothesis graph builder
│   │   ├── belief.py        # Belief update engine
│   │   └── simulator.py     # Monte Carlo simulator
│   ├── schemas/             # Pydantic v2 schemas
│   │   └── __init__.py      # All schema definitions
│   ├── storage/             # Database layer
│   │   ├── models.py        # SQLAlchemy models
│   │   └── database.py      # Connection management
│   ├── config.py            # Application configuration
│   └── main.py              # FastAPI app entry point
├── alembic/                 # Database migrations
│   └── versions/            # Migration files
├── data/seeds/              # Domain seed data
│   ├── pds_biometric_v1.json  # PDS policy hypotheses
│   └── loader.py            # Seed data utilities
├── tests/                   # Test suite
│   ├── test_graph.py        # Graph builder tests
│   ├── test_belief.py       # Belief engine tests
│   └── test_simulator.py    # Simulator tests
├── pyproject.toml           # Dependencies and config
└── alembic.ini              # Alembic configuration
```

## Design Patterns

### 1. Layered Architecture

**API Layer** (`app/api/v1/`)
- FastAPI route handlers only
- No business logic
- Input validation via Pydantic
- Calls engine + storage

**Engine Layer** (`app/engine/`)
- Pure Python business logic
- NO FastAPI dependencies
- Deterministic, testable functions
- Stateless operations

**Storage Layer** (`app/storage/`)
- SQLAlchemy models
- Database operations
- Transaction management

**Schema Layer** (`app/schemas/`)
- Pydantic v2 models
- Contract between layers
- Validation rules

### 2. Separation of Concerns

```python
# ❌ BAD: Business logic in API route
@router.post("/simulate")
def simulate(run_id: UUID):
    # ... graph building logic ...
    # ... belief updates ...
    # ... simulation ...
    return result

# ✅ GOOD: API delegates to engine
@router.post("/simulate")
def simulate(run_id: UUID, db: Session):
    hypotheses = get_hypotheses(db, run_id)
    beliefs = get_latest_beliefs(db, run_id)
    graph = get_graph(db, run_id)
    
    # Engine does the work
    simulator = MonteCarloSimulator(hypotheses, beliefs, graph)
    result = simulator.run_simulation(run_id)
    
    # Store result
    save_simulation(db, result)
    return result
```

### 3. Schema-Driven Development

All data flows through Pydantic schemas:

```python
# Storage → Schema → API
db_hypothesis = db.query(HypothesisModel).first()
hypothesis = Hypothesis.model_validate(db_hypothesis)
return hypothesis  # FastAPI serializes automatically

# API → Schema → Storage
hypothesis_data = HypothesisCreate(**request.json())
db_hypothesis = HypothesisModel(**hypothesis_data.model_dump())
db.add(db_hypothesis)
```

## Adding New Features

### Adding a New Hypothesis Field

1. **Update Schema** (`app/schemas/__init__.py`):
```python
class HypothesisBase(BaseModel):
    # ... existing fields ...
    new_field: str = Field(..., description="Description")
```

2. **Update Model** (`app/storage/models.py`):
```python
class Hypothesis(Base):
    # ... existing columns ...
    new_field = Column(String(500), nullable=False)
```

3. **Create Migration**:
```bash
poetry run alembic revision --autogenerate -m "Add new_field to hypotheses"
poetry run alembic upgrade head
```

4. **Update Seed Data** (`data/seeds/*.json`):
```json
{
  "hid": "H001",
  "new_field": "value",
  ...
}
```

### Adding a New API Endpoint

1. **Create Route** (`app/api/v1/your_module.py`):
```python
from fastapi import APIRouter, Depends
from app.schemas import YourSchema
from app.storage import get_db

router = APIRouter(prefix="/your-resource", tags=["your-resource"])

@router.get("/{id}", response_model=YourSchema)
def get_resource(id: UUID, db: Session = Depends(get_db)):
    # Fetch from DB
    # Convert to schema
    # Return
    pass
```

2. **Register Router** (`app/api/v1/__init__.py`):
```python
from app.api.v1 import your_module

router.include_router(your_module.router)
```

### Adding New Engine Logic

1. **Create Engine Module** (`app/engine/your_logic.py`):
```python
# NO FastAPI imports!
from typing import List, Dict
from app.schemas import Hypothesis

class YourEngine:
    def __init__(self, hypotheses: List[Hypothesis]):
        self.hypotheses = hypotheses
    
    def process(self) -> Dict:
        # Pure Python logic
        return result
```

2. **Add Tests** (`tests/test_your_logic.py`):
```python
import pytest
from app.engine.your_logic import YourEngine

def test_your_engine():
    engine = YourEngine([...])
    result = engine.process()
    assert result["key"] == "expected"
```

3. **Wire to API** (`app/api/v1/your_endpoint.py`):
```python
from app.engine.your_logic import YourEngine

@router.post("/process")
def process(run_id: UUID, db: Session):
    hypotheses = get_hypotheses(db, run_id)
    engine = YourEngine(hypotheses)
    result = engine.process()
    return result
```

## Testing Guidelines

### Unit Tests for Engine

```python
# Test pure logic without database
def test_graph_builder():
    hypotheses = [...]  # Create test data
    builder = HypothesisGraphBuilder(hypotheses)
    builder.add_edge("H001", "H002", EdgeType.REINFORCES)
    
    assert builder.graph.number_of_edges() == 1
```

### Integration Tests for API

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_run():
    response = client.post("/api/v1/runs", json={
        "policy_rule": "Test rule",
        "domain": "TEST"
    })
    assert response.status_code == 201
    assert "id" in response.json()
```

### Test Fixtures

```python
@pytest.fixture
def sample_run(db):
    """Fixture providing a test run with data."""
    run = create_run(db, ...)
    hypotheses = load_hypotheses(db, run.id, ...)
    return run

def test_with_fixture(sample_run):
    assert sample_run.hypothesis_count > 0
```

## Database Best Practices

### Migrations

```bash
# Always review auto-generated migrations
poetry run alembic revision --autogenerate -m "Description"
# Edit alembic/versions/xxx_description.py to verify

# Test migration
poetry run alembic upgrade head
poetry run alembic downgrade -1
poetry run alembic upgrade head
```

### Transactions

```python
# Use context manager for transactions
with db.begin():
    run = Run(...)
    db.add(run)
    hypotheses = [...]
    db.add_all(hypotheses)
# Automatic commit/rollback
```

### Efficient Queries

```python
# ❌ BAD: N+1 queries
for run in runs:
    hypotheses = db.query(Hypothesis).filter_by(run_id=run.id).all()

# ✅ GOOD: Join or eager load
from sqlalchemy.orm import joinedload
runs = db.query(Run).options(joinedload(Run.hypotheses)).all()
```

## Performance Considerations

### Simulation Scaling

- Default: 1000 trajectories, 10 steps
- Scales linearly with n_trajectories
- Use `random_seed` for reproducibility
- Consider async execution for large simulations

### Database JSONB Indexing

```python
# Add GIN index for JSONB queries
op.create_index(
    'ix_hypotheses_stakeholders_gin',
    'hypotheses',
    ['stakeholders'],
    postgresql_using='gin'
)
```

### Caching Considerations

- Belief states are time-indexed (no caching needed)
- Graph structure could be cached per run
- Simulation results are immutable (safe to cache)

## Logging and Debugging

### Enable SQL Logging

```python
# In app/storage/database.py
engine = create_engine(
    settings.database_url,
    echo=True,  # Logs all SQL queries
)
```

### Add Custom Logging

```python
import logging

logger = logging.getLogger(__name__)

def process():
    logger.info("Starting process")
    logger.debug(f"Parameters: {params}")
    try:
        result = compute()
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Process failed: {e}", exc_info=True)
    return result
```

## Common Pitfalls

### 1. Importing FastAPI in Engine

```python
# ❌ BAD
from fastapi import HTTPException
from app.engine.graph import HypothesisGraphBuilder

# ✅ GOOD
# Engine raises ValueError, API converts to HTTPException
try:
    builder.add_edge(...)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
```

### 2. Not Using Schemas

```python
# ❌ BAD: Raw dict manipulation
hypothesis = {"hid": "H001", "stakeholders": [...]}

# ✅ GOOD: Schema validation
hypothesis = HypothesisCreate(
    hid="H001",
    stakeholders=[...],
    ...
)
```

### 3. Missing Type Hints

```python
# ❌ BAD
def process(data):
    return result

# ✅ GOOD
def process(data: List[Hypothesis]) -> Dict[str, float]:
    return result
```

## Code Review Checklist

- [ ] Type hints on all functions
- [ ] Pydantic schemas for data validation
- [ ] No business logic in API routes
- [ ] Engine code has no FastAPI imports
- [ ] Tests cover new functionality
- [ ] Database migrations are reversible
- [ ] Documentation updated
- [ ] Logging added for debugging
- [ ] Error handling with proper exceptions

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
