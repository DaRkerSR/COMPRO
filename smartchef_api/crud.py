from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import or_
from fastapi import HTTPException
import hashlib
try:
    import bcrypt
except Exception as e:
    bcrypt = None
from sqlalchemy.exc import IntegrityError

import models
import schemas

# Use bcrypt_sha256 to avoid bcrypt 72-byte input limitation by pre-hashing with SHA256
pwd_context = None


# ================= PASSWORD =================
def get_password_hash(password: str) -> str:
    """Hash password by pre-hashing with SHA256 and then bcrypt.

    This avoids bcrypt's 72-byte input limit by hashing the password
    to a fixed-size digest first.
    Requires the `bcrypt` package to be installed in the environment.
    """
    if bcrypt is None:
        raise RuntimeError("bcrypt package is required for password hashing. Install with 'pip install bcrypt'.")
    if password is None:
        password = ""
    # Pre-hash with SHA256 (binary digest)
    pre = hashlib.sha256(password.encode('utf-8')).digest()
    hashed = bcrypt.hashpw(pre, bcrypt.gensalt())
    # store as utf-8 string
    return hashed.decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    if bcrypt is None:
        raise RuntimeError("bcrypt package is required for password verification. Install with 'pip install bcrypt'.")
    if plain is None:
        plain = ""
    pre = hashlib.sha256(plain.encode('utf-8')).digest()
    try:
        return bcrypt.checkpw(pre, hashed.encode('utf-8'))
    except Exception:
        return False


# ================= USERS =================
def create_user(db: Session, user: schemas.UserCreate):
    hashed_pw = get_password_hash(user.password)

    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_pw,
        role=user.role,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user



# ================= RECIPES =================
def get_recipe(db: Session, recipe_id: int):
    return db.query(models.ResepMakanan).filter(
        models.ResepMakanan.id_resep_makanan == recipe_id
    ).first()


def get_recipes(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.ResepMakanan).offset(skip).limit(limit).all()


def create_recipe_multi(db: Session, payload: schemas.RecipeCreateMulti):
    recipe = models.ResepMakanan(
        jumlah_resep=payload.jumlah_resep,
        nutrisi=payload.nutrisi,
    )
    db.add(recipe)
    db.commit()
    db.refresh(recipe)

    details = []
    for item in payload.items:
        d = models.DetailResep(
            id_resep_makanan=recipe.id_resep_makanan,
            id_bahan=item.id_bahan,
            jumlah=item.jumlah,
            nutrisi=item.nutrisi,
        )
        db.add(d)
        details.append(d)

    db.commit()
    return recipe, details


def get_or_create_bahan_by_name(db: Session, nama_bahan: str, berat: float, jumlah: int):
    bahan = db.query(models.BahanMakan).filter(
        models.BahanMakan.nama_bahan == nama_bahan
    ).first()

    if bahan:
        return bahan

    bahan = models.BahanMakan(
        nama_bahan=nama_bahan,
        berat=berat or 0,
        jumlah=jumlah or 0,
    )

    db.add(bahan)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        bahan = db.query(models.BahanMakan).filter(
            models.BahanMakan.nama_bahan == nama_bahan
        ).first()

    return bahan


def create_recipe_by_name(db: Session, payload: schemas.RecipeCreateByName):
    recipe = models.ResepMakanan(
        jumlah_resep=payload.jumlah_resep,
        nutrisi=payload.nutrisi,
    )
    db.add(recipe)
    db.commit()
    db.refresh(recipe)

    details = []
    for item in payload.items:
        bahan = get_or_create_bahan_by_name(
            db, item.nama_bahan, item.berat, item.jumlah
        )
        d = models.DetailResep(
            id_resep_makanan=recipe.id_resep_makanan,
            id_bahan=bahan.id_bahan,
            bahan_digunakan=bahan.nama_bahan,
            jumlah=item.jumlah,
            nutrisi=item.nutrisi,
        )
        db.add(d)
        details.append(d)

    db.commit()
    return recipe, details


def update_recipe(db: Session, recipe_id: int, recipe_in: schemas.RecipeUpdate):
    recipe = get_recipe(db, recipe_id)
    if not recipe:
        return None

    for k, v in recipe_in.model_dump(exclude_unset=True).items():
        setattr(recipe, k, v)

    db.commit()
    db.refresh(recipe)
    return recipe


def delete_recipe(db: Session, recipe_id: int):
    recipe = get_recipe(db, recipe_id)
    if not recipe:
        return False

    db.delete(recipe)
    db.commit()
    return True
def create_recipe_full(db: Session, recipe_data: schemas.RecipeCreate):
    """
    Create resep LENGKAP dengan judul, steps, loves, URL
    """
    # 1. Buat resep utama
    resep = models.ResepMakanan(
        judul=recipe_data.judul,
        steps=recipe_data.steps,
        url=recipe_data.url,
        loves=recipe_data.loves,
        jumlah_resep=recipe_data.jumlah_resep,
        nutrisi=recipe_data.nutrisi
    )
    db.add(resep)
    db.flush()  # dapat ID dulu
    
    # 2. Buat bahan & detail
    for item in recipe_data.items:
        # Cari atau buat bahan (pakai logic yang sudah ada)
        bahan = get_or_create_bahan_by_name(
            db, item.nama_bahan, 0, item.jumlah
        )
        
        # Detail resep
        detail = models.DetailResep(
            id_resep_makanan=resep.id_resep_makanan,
            id_bahan=bahan.id_bahan,
            bahan_digunakan=item.nama_bahan,
            jumlah=item.jumlah,
            nutrisi=item.nutrisi
        )
        db.add(detail)
    
    db.commit()
    db.refresh(resep)
    return resep


def search_recipe(db: Session, bahan_list: list):
    q = (
        db.query(models.ResepMakanan)
        .join(models.DetailResep)
        .join(models.BahanMakan)
    )

    filters = [
        or_(
            models.BahanMakan.nama_bahan.ilike(f"%{b}%"),
            models.DetailResep.bahan_digunakan.ilike(f"%{b}%"),
        )
        for b in bahan_list
    ]

    return q.filter(or_(*filters)).distinct().all()
# ================= FAVORITES =================
def add_favorite(db: Session, user_id: int, recipe_id: int):
    """Tambah resep ke favorites user"""
    # Cek apakah sudah ada
    existing = db.query(models.FavoriteResep).filter(
        models.FavoriteResep.id_user == user_id,
        models.FavoriteResep.id_resep_makanan == recipe_id
    ).first()
    
    if existing:
        return False  # sudah ada
    
    # Buat baru
    favorite = models.FavoriteResep(
        id_user=user_id,
        id_resep_makanan=recipe_id
    )
    db.add(favorite)
    db.commit()
    db.refresh(favorite)
    return True

def remove_favorite(db: Session, user_id: int, recipe_id: int):
    """Hapus resep dari favorites"""
    favorite = db.query(models.FavoriteResep).filter(
        models.FavoriteResep.id_user == user_id,
        models.FavoriteResep.id_resep_makanan == recipe_id
    ).first()
    
    if not favorite:
        return False
    
    db.delete(favorite)
    db.commit()
    return True

def get_user_favorites(db: Session, user_id: int):
    """Ambil semua favorites user"""
    return db.query(models.FavoriteResep).filter(
        models.FavoriteResep.id_user == user_id
    ).all()
