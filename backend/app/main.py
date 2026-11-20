
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db


# IMPORTANT:
# Initialize database tables BEFORE importing routers.
# This prevents ModelStore from querying model_states
# before the table has been created.

init_db()


# Import application components only after database initialization
from app.services.latency_tracker import latency_tracker
from app.routers import (
    auth,
    products,
    cart,
    wishlist,
    orders,
    recommendations,
    events,
    admin,
)


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multi-category E-commerce Platform with Adaptive LinUCB Recommendation Bandits",
    version="1.0.0",
    lifespan=lifespan,
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# Request timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000.0

    latency_tracker.record_e2e(request.url.path, duration_ms)

    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

    return response


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "https://smartcart-ai-steel.vercel.app",
],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(auth.router)
app.include_router(products.router)
app.include_router(cart.router)
app.include_router(wishlist.router)
app.include_router(orders.router)
app.include_router(recommendations.router)
app.include_router(events.router)
app.include_router(admin.router)


@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
