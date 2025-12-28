from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import database
from database import Base, engine
from database import get_db  # Pastikan ada

app = FastAPI(title="SmartChef API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()
Base.metadata.create_all(bind=engine)

# DIRECT IMPORT - NO __init__.py drama
from routers.recipes import router as recipes_router
from routers.auth import router as auth_router
from routers.favorites import router as favorites_router

# Include routers
app.include_router(recipes_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(favorites_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "SmartChef API 🚀 - Recipes + Favorites LIVE!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
