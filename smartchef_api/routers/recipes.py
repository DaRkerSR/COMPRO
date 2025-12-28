from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, desc
import models
import schemas
from database import get_db
import crud

router = APIRouter(prefix="/recipes", tags=["recipes"])

# 🔥 1. ADVANCED SEARCH (Cookpad Style)
@router.get("/search")
def search_recipes(
    bahan: Optional[str] = Query(None, description="ayam,bawang,cabe"),
    kategori: Optional[str] = Query(None, description="ayam,sayur,manis"),
    waktu_max: Optional[int] = Query(None, description="menit, ex: 30"),
    kesulitan: Optional[str] = Query(None, regex="^(mudah|sedang|sulit)$"),
    sort: str = Query("loves", regex="^(loves|recent|random)$"),
    page: int = Query(1, ge=1),
    limit: int = Query(24, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Cookpad Advanced Search!
    GET /recipes/search?bahan=ayam&sort=loves&page=1&limit=24
    """
    offset = (page - 1) * limit
    
    # Base query
    query = db.query(models.ResepMakanan).join(models.DetailResep).join(models.BahanMakan)
    
    # Filter bahan
    if bahan:
        bahan_list = [b.strip().lower() for b in bahan.split(",")]
        conditions = []
        for b in bahan_list:
            conditions.extend([
                models.BahanMakan.nama_bahan.ilike(f"%{b}%"),
                models.DetailResep.bahan_digunakan.ilike(f"%{b}%")
            ])
        query = query.filter(or_(*conditions))
    
    # Filter kategori (judul)
    if kategori:
        query = query.filter(models.ResepMakanan.judul.ilike(f"%{kategori}%"))
    
    # Sort
    if sort == "loves":
        query = query.order_by(desc(models.ResepMakanan.loves))
    elif sort == "recent":
        query = query.order_by(desc(models.ResepMakanan.created_at))
    elif sort == "random":
        query = query.order_by(func.rand())
    
    recipes = query.group_by(models.ResepMakanan.id_resep_makanan)\
                   .offset(offset).limit(limit).all()
    
    total = db.query(func.count(models.ResepMakanan.id_resep_makanan)).join(models.DetailResep).join(models.BahanMakan).first()[0]
    
    return {
        "results": [
            {
                "id_resep_makanan": r.id_resep_makanan,
                "judul": getattr(r, 'judul', f'Resep #{r.id_resep_makanan}'),
                "loves": getattr(r, 'loves', 0),
                "url": f"/recipes/{r.id_resep_makanan}/detail",
                "preview": getattr(r, 'steps', '')[:100] + "..."
            }
            for r in recipes
        ],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        },
        "filters": {
            "bahan": bahan,
            "sort": sort
        }
    }

# 🔥 2. POPULAR RECIPES (Homepage)
@router.get("/popular")
def get_popular_recipes(limit: int = Query(10, le=50), db: Session = Depends(get_db)):
    """Homepage trending recipes"""
    recipes = db.query(models.ResepMakanan)\
                .order_by(desc(models.ResepMakanan.loves))\
                .limit(limit).all()
    
    return [
        {
            "id_resep_makanan": r.id_resep_makanan,
            "judul": getattr(r, 'judul', 'Resep Populer'),
            "loves": getattr(r, 'loves', 0),
            "url": f"/recipes/{r.id_resep_makanan}/detail"
        }
        for r in recipes
    ]

# 🔥 3. RANDOM RECIPES (Inspirasi)
@router.get("/random")
def get_random_recipes(limit: int = Query(5, le=20), db: Session = Depends(get_db)):
    """Inspirasi resep random"""
    recipes = db.query(models.ResepMakanan)\
                .order_by(func.rand())\
                .limit(limit).all()
    
    return [
        {
            "id_resep_makanan": r.id_resep_makanan,
            "judul": getattr(r, 'judul', 'Resep Random'),
            "loves": getattr(r, 'loves', 0),
            "url": f"/recipes/{r.id_resep_makanan}/detail"
        }
        for r in recipes
    ]

# ✅ 4. CREATE RESEP (tetap sama)
@router.post("/", response_model=schemas.RecipeHeaderOut)
def create_recipe(recipe_data: schemas.RecipeCreate, db: Session = Depends(get_db)):
    """
    Buat resep baru - langsung muncul di search!
    """
    resep = crud.create_recipe_full(db, recipe_data)
    return schemas.RecipeHeaderOut(
        id_resep_makanan=resep.id_resep_makanan,
        judul=resep.judul,
        loves=resep.loves,
        url=f"/recipes/{resep.id_resep_makanan}/detail"
    )

# ✅ 5. DETAIL RESEP (upgrade)
@router.get("/{recipe_id}/detail")
def get_recipe_detail(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(models.ResepMakanan).filter(
        models.ResepMakanan.id_resep_makanan == recipe_id
    ).first()
    
    if not recipe:
        raise HTTPException(status_code=404, detail="Resep tidak ditemukan")
    
    ingredients = db.query(models.DetailResep).filter(
        models.DetailResep.id_resep_makanan == recipe_id
    ).all()
    
    return {
        "id_resep_makanan": recipe.id_resep_makanan,
        "judul": getattr(recipe, 'judul', 'Judul tidak tersedia'),
        "steps": getattr(recipe, 'steps', 'Langkah tidak tersedia'),
        "loves": getattr(recipe, 'loves', 0),
        "url": getattr(recipe, 'url', None),
        "created_at": getattr(recipe, 'created_at', None),
        "ingredients": [
            {
                "bahan": d.bahan_digunakan or "tidak diketahui",
                "jumlah": getattr(d, 'jumlah', 1),
                "nutrisi": getattr(d, 'nutrisi', None)
            }
            for d in ingredients
        ]
    }
