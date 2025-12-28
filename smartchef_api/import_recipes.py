"""
import_recipes.py
Script untuk import CSV ke database SmartChef.

Jalankan:
    python import_recipes.py
"""

import csv
import sys
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Import dari project kamu
from database import SessionLocal, engine
import models

# Pastikan database sudah siap
models.Base.metadata.create_all(bind=engine)


def normalize_ingredient_name(name: str) -> str:
    """
    Normalisasi nama bahan:
    - strip whitespace
    - lowercase
    - buang angka di depan (misal "7 Bawang Merah" -> "bawang merah")
    """
    # Buang karakter whitespace extra
    name = name.strip()
    
    # Split dan buang angka/unit di depan
    parts = name.split()
    result = []
    started = False
    
    for part in parts:
        # skip angka, pecahan, unit umum di depan
        if not started and part[0].isdigit() or part in ["sdm", "sdt", "kg", "g", "ml", "butir", "biji", "buah", "lembar", "batang", "ikat", "siung", "ruas", "ekor", "gelas", "sendok"]:
            continue
        started = True
        result.append(part)
    
    return " ".join(result).lower() if result else name.lower()


def parse_ingredients(ingredients_str: str) -> list[str]:
    """
    Parse bahan dari format CSV (pemisah: --)
    Return: list of ingredient names (normalized)
    """
    if not ingredients_str:
        return []
    
    # Split by -- (pemisah di CSV)
    raw_items = ingredients_str.split("--")
    
    ingredients = []
    for item in raw_items:
        item = item.strip()
        if item:  # jangan kosong
            normalized = normalize_ingredient_name(item)
            if normalized:
                ingredients.append(normalized)
    
    return ingredients


def get_or_create_bahan(db: Session, nama_bahan: str) -> models.BahanMakan:
    """
    Cari atau buat bahan baru (upsert).
    """
    normalized = nama_bahan.strip().lower()
    
    bahan = db.query(models.BahanMakan).filter(
        models.BahanMakan.nama_bahan == normalized
    ).first()
    
    if bahan:
        return bahan
    
    # Buat baru
    bahan = models.BahanMakan(nama_bahan=normalized)
    db.add(bahan)
    
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # Kemungkinan sudah dibuat thread lain, cari lagi
        bahan = db.query(models.BahanMakan).filter(
            models.BahanMakan.nama_bahan == normalized
        ).first()
    
    return bahan


def import_recipes_from_csv(csv_path: str, limit: int = None):
    """
    Import resep dari CSV ke database.
    
    Args:
        csv_path: path ke file CSV
        limit: jumlah baris max untuk import (None = semua)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ File tidak ditemukan: {csv_path}")
        return
    
    db = SessionLocal()
    count_imported = 0
    count_failed = 0
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                if limit and idx >= limit:
                    break
                
                try:
                    title = row.get("Title", "").strip()
                    ingredients_raw = row.get("Ingredients", "")
                    steps = row.get("Steps", "").strip()
                    url = row.get("URL", "").strip()
                    loves = int(row.get("Loves", 0)) if row.get("Loves") else 0
                    
                    # Validasi minimal
                    if not title or not ingredients_raw:
                        count_failed += 1
                        continue
                    
                    # 1. Buat ResepMakanan
                    resep = models.ResepMakanan(
                        judul=title,
                        steps=steps,
                        url=url,
                        loves=loves,
                        jumlah_resep=1,
                        nutrisi=""  # bisa diisi nanti
                    )
                    db.add(resep)
                    db.flush()  # jadi dapat id_resep_makanan
                    
                    # 2. Parse ingredients dan buat DetailResep
                    ingredients_list = parse_ingredients(ingredients_raw)
                    
                    for ing_name in ingredients_list:
                        # Dapatkan atau buat bahan
                        bahan = get_or_create_bahan(db, ing_name)
                        
                        # Buat detail_resep
                        detail = models.DetailResep(
                            id_resep_makanan=resep.id_resep_makanan,
                            id_bahan=bahan.id_bahan,
                            bahan_digunakan=ing_name,
                            jumlah=1,
                            nutrisi=""
                        )
                        db.add(detail)
                    
                    db.commit()
                    count_imported += 1
                    
                    # Progress indicator
                    if (idx + 1) % 100 == 0:
                        print(f"✓ Imported {idx + 1} resep...")
                
                except Exception as e:
                    db.rollback()
                    count_failed += 1
                    print(f"⚠️ Error di baris {idx + 1}: {e}")
                    continue
        
        print(f"\n{'='*50}")
        print(f"✅ Import selesai!")
        print(f"   Berhasil: {count_imported}")
        print(f"   Gagal: {count_failed}")
        print(f"   Total ingredients unik: {db.query(models.BahanMakan).count()}")
        print(f"{'='*50}")
        
    finally:
        db.close()


if __name__ == "__main__":
    csv_file = "all_cleaned_data.csv"  # sesuaikan nama file
    
    print(f"📥 Mulai import dari: {csv_file}")
    print(f"⏳ Ini bisa memakan waktu beberapa menit...")
    
    # Uncomment untuk test dengan 100 baris dulu:
    # import_recipes_from_csv(csv_file, limit=100)
    
    # Uncomment untuk import semua:
    import_recipes_from_csv(csv_file)
