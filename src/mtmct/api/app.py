"""FastAPI application (spec §10.1). Binds to 127.0.0.1 via `mtmct api`; CORS only for the Vite dev server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from mtmct.api import datasets, jobs, media, runs, system, views
from mtmct.config import paths
from mtmct.db import get_engine

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    get_engine()  # creates db/mtmct.sqlite and the schema
    warm = asyncio.create_task(asyncio.to_thread(system.warmup))
    yield
    warm.cancel()


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="MTMCT PoC", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    for module in (system, datasets, runs, views, jobs, media):
        app.include_router(module.router, prefix="/api")

    dist = static_dir if static_dir is not None else paths().root / "web" / "dist"
    if (dist / "index.html").exists():
        root = dist.resolve()

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(404)
            candidate = (root / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(candidate)
            return FileResponse(root / "index.html")

    return app


app = create_app()
