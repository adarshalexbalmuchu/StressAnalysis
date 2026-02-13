# PDS Stress Test Engine - V1

Production-grade pre-implementation policy stress-testing system for analyzing the impact of mandatory Aadhaar-based biometric authentication at Fair Price Shops in India's Public Distribution System.

## Core Philosophy

This is **NOT** a recommendation engine. It:
- Models multiple plausible futures under uncertainty
- Maintains explicit probabilistic belief states
- Does NOT score, rank, or automate policy decisions
- Focuses on mechanistic hypotheses about stakeholder adaptations

## Architecture

```
pds-stress-test/
├── app/
│   ├── main.py             # FastAPI application entry point
│   ├── config.py           # Application settings
│   ├── api/v1/
│   │   ├── __init__.py     # Router aggregation
│   │   └── router.py       # All API endpoints (runs + pipeline)
│   ├── engine/
│   │   ├── graph.py        # Hypothesis graph (build, validate, class API)
│   │   ├── belief.py       # Bayesian belief update engine
│   │   └── simulator.py    # Monte Carlo trajectory simulation
│   ├── schemas/            # Pydantic v2 schemas (contract layer)
│   └── storage/            # SQLAlchemy models + database layer
├── alembic/                # Database migrations (single 001_initial_schema)
├── data/seeds/             # Domain-specific seed data
├── docs/                   # Technical deep-dive documentation
├── tests/                  # Test suite
└── pyproject.toml          # Dependencies and configuration
```

## Documentation

For technical deep-dives, see the [docs/](docs/) folder:
- [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Detailed project layout
- [DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development guide
- [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) - Implementation notes
- [VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md) - Verification results

## Tech Stack

- **Python 3.11+**
- **FastAPI** - API interface layer
- **Pydantic v2** - Schema validation and contracts
- **SQLAlchemy 2.0** - ORM and database abstraction
- **Alembic** - Database migrations
- **PostgreSQL** - Primary persistence (with JSONB for artifacts)
- **NetworkX** - Hypothesis graph management
- **NumPy** - Monte Carlo simulation

## Domain Model

### Hypothesis
A mechanistic adaptation claim with structure:
- Trigger → Stakeholder response → Downstream effects
- Contains: stakeholders, triggers, mechanism, effects, time_horizon

### Hypothesis Graph
- Nodes = hypotheses
- Edges = relationships (reinforces, contradicts, depends_on)

### Belief State
- Maintains P(H_i) for each hypothesis
- Updates via rule-based Bayesian logic (no LLMs)
- Includes explanation logs

### Signals
- Weak, noisy evidence (grievances, circulars, media, audits)
- Affects beliefs probabilistically

### Trajectories
- Coherent clusters of activated hypotheses
- Generated via Monte Carlo simulation
- Output = distribution over trajectories

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL
- Poetry (or pip)

### Installation

```bash
# Install dependencies
poetry install

# Setup database
createdb pds_stress_test

# Run migrations
poetry run alembic upgrade head

# Start development server
poetry run uvicorn app.main:app --reload
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Principles

- ✅ Type hints everywhere
- ✅ No global state
- ✅ Clear function boundaries
- ✅ Deterministic, reproducible behavior
- ✅ Clean logging for explainability
- ❌ No LLMs for belief updates or scoring
- ❌ No automation of policy decisions

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app tests/
```

## V1 Scope

This version implements:
1. Run creation and management
2. Hypothesis seeding (PDS biometric authentication domain)
3. Hypothesis graph construction
4. Rule-based belief state initialization and updates
5. Monte Carlo trajectory simulation
6. Full persistence layer with migrations

**Out of scope for V1:**
- LLM integration for hypothesis generation
- UI beyond Swagger
- Real-time signal ingestion
- Multi-policy comparison

## License

Proprietary - Policy Analysis Team
