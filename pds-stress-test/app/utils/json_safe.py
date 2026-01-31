"""
JSON-safe serialization utility.

Recursively converts non-JSON-serializable types to JSON-safe primitives.
Essential for storing complex objects in Postgres JSONB fields.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID


def json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable primitives.
    Safe for storing into Postgres JSONB.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [json_safe(v) for v in sorted(obj)]
    # Pydantic v2 objects
    if hasattr(obj, "model_dump"):
        return json_safe(obj.model_dump(mode="json"))
    # SQLAlchemy ORM objects sometimes slip in
    if hasattr(obj, "__dict__"):
        # last resort: stringify
        return str(obj)
    return str(obj)
