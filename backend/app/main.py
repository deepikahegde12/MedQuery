"""FastAPI entry point. Run from the `backend/` directory:

    uvicorn app.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import PROJECT_DIR, Settings, get_settings
from app.container import Services, build_services
from app.routes import router
from app.schemas import error

logger = logging.getLogger("medquery")
FRONTEND_DIR = PROJECT_DIR / "frontend"


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(settings)
        svc = app.state.services
        if settings.preload_models:
            svc.encoder.model.warm_up_in_background()
            if hasattr(svc.generator, "model"):
                svc.generator.model.warm_up_in_background()
        logger.info("MedQuery started (knowledge base configured: %s)", svc.retrieval.is_configured)
        yield
        svc.close()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            length = request.headers.get("content-length")
            body = await request.body() if length is None else None
            size = int(length) if length and length.isdigit() else len(body or b"")
            if size > settings.max_request_bytes:
                return JSONResponse(status_code=413, content=error(
                    "payload_too_large", f"Request body exceeds {settings.max_request_bytes} bytes."))
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in e.get("loc", ())[1:]) or "body",
             "message": str(e.get("msg", "")).removeprefix("Value error, ")}
            for e in exc.errors()
        ]
        message = details[0]["message"] if details else "Invalid request."
        return JSONResponse(status_code=422, content=error("validation_error", message, details))

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content=error("http_error", str(exc.detail)))

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=error(
            "internal_error", "Something went wrong while processing the request. Please retry."))

    app.include_router(router)

    # Serve the frontend explicitly (not a catch-all mount at "/") so unknown or
    # wrong-method /api requests still get JSON 404/405 responses.
    if FRONTEND_DIR.is_dir():
        for sub in ("css", "js"):
            if (FRONTEND_DIR / sub).is_dir():
                app.mount(f"/{sub}", StaticFiles(directory=FRONTEND_DIR / sub), name=sub)

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
