from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.demo_guard import DemoGuard, demo_mode_enabled

app = FastAPI(title="MedOrchestrate")

# Vite's default dev server port is 5173 (confirmed: frontend/package.json
# has no vite.config port override, so it runs on Vite's default);
# 3000 is also allowed since that's the common CRA/Next.js default, in
# case the frontend setup changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Public-demo protections (rate limit, total case cap, upload size). Off unless DEMO_MODE=on; see backend/demo_guard.py.
# Added last so it is the outermost layer: refused requests never reach the pipeline.
if demo_mode_enabled():
    app.add_middleware(DemoGuard)


@app.get("/health")
def health():
    return {"status": "ok"}
