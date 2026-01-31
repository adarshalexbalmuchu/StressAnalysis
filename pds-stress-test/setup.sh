#!/bin/bash
# Setup script for PDS Stress Test Engine

set -e

echo "🚀 Setting up PDS Stress Test Engine..."

# Check for Python 3.11+
echo "✓ Checking Python version..."
python3 --version | grep -q "Python 3.1[1-9]" || {
    echo "❌ Python 3.11+ required"
    exit 1
}

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Check for PostgreSQL
echo "✓ Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "  PostgreSQL found"
else
    echo "⚠️  PostgreSQL not found. Please install PostgreSQL."
    echo "  Ubuntu/Debian: sudo apt-get install postgresql"
    echo "  macOS: brew install postgresql"
fi

# Create database
echo "🗄️  Setting up database..."
read -p "Create database 'pds_stress_test'? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    createdb pds_stress_test 2>/dev/null || echo "  Database may already exist"
fi

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "  ⚠️  Please edit .env with your database credentials"
fi

# Run migrations
echo "🔄 Running database migrations..."
poetry run alembic upgrade head

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your database credentials"
echo "  2. Run the development server:"
echo "     poetry run uvicorn app.main:app --reload"
echo "  3. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "To load seed data, use the API endpoints or:"
echo "  poetry run python data/seeds/loader.py"
