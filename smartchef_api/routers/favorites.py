from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db  # Pakai global get_db
import models
import crud

router = APIRouter(prefix="/favorites", tags=["Favorites"])

@router.post("/{username}/{recipe_id}")
def add_favorite(username: str, recipe_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    success = crud.add_favorite(db, user.id_user, recipe_id)
    if success:
        return {"message": "Added to favorites", "username": username, "recipe_id": recipe_id}
    return {"message": "Already in favorites"}


@router.post("/by-name/{username}")
class TitlePayload(BaseModel):
    judul: str


def add_favorite_by_name(username: str, payload: TitlePayload, db: Session = Depends(get_db)):
    """Add favorite by recipe title (judul) — accepts JSON {"judul": "..."}"""
    # debug log payload to help track 422 issues
    try:
        print(f"[favorites] add_by_name payload for {username}: {payload.json()}")
    except Exception:
        print(f"[favorites] add_by_name received non-serializable payload for {username}")

    judul = payload.judul
    if not judul:
        raise HTTPException(status_code=400, detail="Missing 'judul' in payload")

    # try to find recipe by exact or partial title
    recipe = db.query(models.ResepMakanan).filter(models.ResepMakanan.judul.ilike(f"%{judul}%"))\
        .order_by(models.ResepMakanan.loves.desc()).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Resep tidak ditemukan")

    success = crud.add_favorite(db, db.query(models.User).filter(models.User.username == username).first().id_user, recipe.id_resep_makanan)
    if success:
        return {"message": "Added to favorites", "username": username, "recipe_id": recipe.id_resep_makanan}
    return {"message": "Already in favorites"}

@router.delete("/{username}/{recipe_id}")
def remove_favorite(username: str, recipe_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    success = crud.remove_favorite(db, user.id_user, recipe_id)
    if success:
        return {"message": "Removed from favorites"}
    raise HTTPException(status_code=404, detail="Favorite not found")

@router.get("/{username}")
def get_favorites(username: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    favs = crud.get_user_favorites(db, user.id_user)
    
    result = []
    for f in favs:
        # FIX: Ambil judul resep + 1 bahan pertama
        recipe = f.resep
        detail = db.query(models.DetailResep).filter(
            models.DetailResep.id_resep_makanan == recipe.id_resep_makanan
        ).first()
        
        result.append({
            "id_favorit_resep": f.id_favorit_resep,
            "username": username,
            "judul_resep": getattr(recipe, "judul", "Unknown Recipe"),
            "loves": getattr(recipe, "loves", 0),
            "bahan": getattr(detail, "bahan_digunakan", "Unknown") if detail else "No ingredients"
        })
    
    return {"favorites": result}
