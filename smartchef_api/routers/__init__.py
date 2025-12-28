# routers/__init__.py (BARU - clean)
from .recipes import router as recipes
from .auth import router as auth
from .favorites import router as favorites
# ❌ NO ingredients!

__all__ = ["recipes", "auth", "favorites"]
