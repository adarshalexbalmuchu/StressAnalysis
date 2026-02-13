# Project Structure

```
pds-stress-test/
│
├── 📋 Documentation
│   ├── README.md                    # Project overview and philosophy
│   ├── QUICKSTART.md                # Quick start guide
│   ├── DEVELOPMENT.md               # Development guide
│   └── IMPLEMENTATION_SUMMARY.md    # Complete implementation report
│
├── ⚙️ Configuration
│   ├── .devcontainer/
│   │   └── devcontainer.json        # GitHub Codespaces config
│   ├── .env.example                 # Environment template
│   ├── .gitignore                   # Git ignore rules
│   ├── pyproject.toml               # Dependencies and config
│   ├── alembic.ini                  # Alembic configuration
│   └── setup.sh                     # Setup automation script
│
├── 🗄️ Database Migrations
│   └── alembic/
│       ├── env.py                   # Alembic environment
│       ├── script.py.mako           # Migration template
│       └── versions/
│           └── 001_initial_schema.py  # Initial database schema
│
├── 🏗️ Application Code
│   └── app/
│       ├── __init__.py
│       ├── config.py                # Application settings
│       ├── main.py                  # FastAPI entry point
│       │
│       ├── 📜 schemas/              # Pydantic V2 Schemas (Contract Layer)
│       │   └── __init__.py          # All schema definitions
│       │                            # - Hypothesis, Graph, BeliefState
│       │                            # - Signal, Trajectory, Run
│       │                            # - All enums and validators
│       │
│       ├── 🗃️ storage/              # Database Layer
│       │   ├── __init__.py          # Storage exports
│       │   ├── database.py          # DB connection & sessions
│       │   └── models.py            # SQLAlchemy models
│       │                            # - Run, Hypothesis, Graph
│       │                            # - BeliefState, Signal, Simulation
│       │
│       ├── 🔧 engine/               # Core Logic (NO FastAPI imports)
│       │   ├── __init__.py          # Engine exports
│       │   ├── graph.py             # HypothesisGraphBuilder
│       │   │                        # - NetworkX graph construction
│       │   │                        # - Validation, centrality, chains
│       │   ├── belief.py            # BeliefUpdateEngine
│       │   │                        # - Bayesian belief updates
│       │   │                        # - Signal integration (NO LLMs)
│       │   │                        # - Graph constraint application
│       │   └── simulator.py         # MonteCarloSimulator
│       │                            # - Trajectory generation
│       │                            # - Cascade propagation
│       │                            # - Sensitivity analysis
│       │
│       └── 🌐 api/v1/               # FastAPI Routes (Interface Layer)
│           ├── __init__.py          # Router aggregation
│           ├── runs.py              # Run CRUD endpoints
│           ├── hypotheses.py        # Hypothesis endpoints
│           ├── graph.py             # Graph construction endpoints
│           ├── beliefs.py           # Belief state endpoints
│           ├── signals.py           # Signal ingestion endpoints
│           └── simulation.py        # Simulation execution endpoints
│
├── 📊 Domain Data
│   └── data/seeds/
│       ├── pds_biometric_v1.json    # PDS seed data (20 hypotheses)
│       └── loader.py                # Seed data utilities
│
├── 🧪 Tests
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py              # Test configuration
│       ├── test_graph.py            # Graph builder tests (12 tests)
│       ├── test_belief.py           # Belief engine tests (11 tests)
│       └── test_simulator.py        # Simulator tests (8 tests)
│
└── 📚 Examples
    └── examples/
        ├── __init__.py
        └── complete_workflow.py     # End-to-end example script
```

## File Count Summary

- **Python files**: 26
- **Configuration files**: 6
- **Documentation files**: 4
- **Data files**: 1 JSON seed
- **Total tracked files**: 37

## Lines of Code (Approximate)

- **Core Engine**: ~1,200 LOC
- **API Routes**: ~800 LOC
- **Schemas**: ~500 LOC
- **Storage**: ~300 LOC
- **Tests**: ~800 LOC
- **Examples**: ~400 LOC
- **Total**: ~4,000 LOC

## Key Files by Purpose

### Must-Read Files (Understanding the System)
1. `README.md` - Start here for philosophy
2. `QUICKSTART.md` - Get running quickly
3. `app/schemas/__init__.py` - Understand data structures
4. `examples/complete_workflow.py` - See it in action

### Core Implementation Files
1. `app/engine/graph.py` - Hypothesis graph logic
2. `app/engine/belief.py` - Belief update logic
3. `app/engine/simulator.py` - Monte Carlo logic
4. `app/storage/models.py` - Database schema
5. `alembic/versions/001_initial_schema.py` - Migration

### Integration Files
1. `app/main.py` - FastAPI application
2. `app/api/v1/*.py` - API endpoints
3. `app/storage/database.py` - DB connection

### Configuration Files
1. `.env.example` - Environment template
2. `pyproject.toml` - Dependencies
3. `alembic.ini` - Migration config
4. `.devcontainer/devcontainer.json` - Codespaces

### Data Files
1. `data/seeds/pds_biometric_v1.json` - Domain hypotheses
2. `data/seeds/loader.py` - Loading utilities

### Quality Assurance Files
1. `tests/test_*.py` - Test suite
2. `setup.sh` - Setup automation
