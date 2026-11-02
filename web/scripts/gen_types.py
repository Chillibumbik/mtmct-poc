"""Print one JSON Schema (with `$defs`) covering every pydantic model of `mtmct.api.schemas`.

Used by `pnpm gen:api` when `web/openapi.json` is absent: `uv run python web/scripts/gen_types.py`.
"""

from __future__ import annotations

import inspect
import json
import sys

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from mtmct.api import schemas


def main() -> None:
    models = [
        obj
        for _, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj is not BaseModel and obj.__module__ == schemas.__name__
    ]
    _, top = models_json_schema([(m, "validation") for m in models], ref_template="#/$defs/{model}")
    defs = top["$defs"]
    for name, schema in defs.items():
        # Responses always carry every field (defaults included); only request bodies have optional keys.
        if not name.endswith(("Create", "In")) and "properties" in schema:
            schema["required"] = list(schema["properties"])
    root = {
        "title": "ApiSchemas",
        "type": "object",
        "properties": {name: {"$ref": f"#/$defs/{name}"} for name in sorted(defs)},
        "$defs": defs,
    }
    json.dump(root, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
