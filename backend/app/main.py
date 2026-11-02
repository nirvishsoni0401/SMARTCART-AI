import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.services.latency_tracker import latency_tracker
from app.routers import auth, products, cart, wishlist, orders, recommendations, events, admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    init_db()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multi-category E-commerce Platform with Adaptive LinUCB Recommendation Bandits",
    version="1.0.0",
    lifespan=lifespan
)

# Timing middleware for end-to-end endpoint latency tracking
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    latency_tracker.record_e2e(request.url.path, duration_ms)
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response

# CORS middleware for frontend communication (:5173, :3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core e-commerce, recommendation & admin routers
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
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
