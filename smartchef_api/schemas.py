from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

# ================= USERS =================
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "user"


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id_user: int
    username: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True



# ================= RECIPES =================

# ---- Multi create (pakai id_bahan) ----
class RecipeItemCreate(BaseModel):
    id_bahan: int
    jumlah: int
    nutrisi: Optional[str] = None


class RecipeCreateMulti(BaseModel):
    jumlah_resep: int
    nutrisi: str
    items: List[RecipeItemCreate]


class RecipeItemOut(BaseModel):
    id_detail_resep: int
    id_bahan: int
    jumlah: int
    nutrisi: Optional[str]

    class Config:
        from_attributes = True


class RecipeOutMulti(BaseModel):
    id_resep_makanan: int
    jumlah_resep: int
    nutrisi: str
    items: List[RecipeItemOut]

    class Config:
        from_attributes = True


# ---- Multi create (pakai nama_bahan) ----
class RecipeItemByName(BaseModel):
    nama_bahan: str
    jumlah: int
    berat: Optional[float] = None
    nutrisi: Optional[str] = None


class RecipeCreateByName(BaseModel):
    jumlah_resep: int
    nutrisi: str
    items: List[RecipeItemByName]


# ---- Output ----
class RecipeHeaderOut(BaseModel):
    
    id_resep_makanan: int
    judul : str
    loves : int
    url: Optional[str] = None


    class Config:
        from_attributes = True

class RecipeDetailOut(BaseModel):
    id_resep_makanan: int
    judul: Optional[str] = None          # ← OPTIONAL
    steps: Optional[str] = None          # ← OPTIONAL  
    url: Optional[str] = None
    loves: Optional[int] = 0             # ← OPTIONAL
    created_at: Optional[datetime] = None # ← OPTIONAL
    jumlah_resep: int = 1
    nutrisi: Optional[str] = None

    class Config:
        from_attributes = True

class RecipeItemDetailOut(BaseModel):
    id_detail_resep: int
    id_bahan: Optional[int] = None
    bahan_digunakan: str
    jumlah: int
    nutrisi: Optional[str] = None

    class Config:
        from_attributes = True


class RecipeWithItemsOut(RecipeDetailOut):
    items: List[RecipeItemDetailOut]

    class Config:
        from_attributes = True


class RecipeUpdate(BaseModel):
    jumlah_resep: Optional[int] = None
    nutrisi: Optional[str] = None
# ================= RECIPE CREATE (CRITICAL!) =================
class IngredientCreate(BaseModel):
    nama_bahan: str
    jumlah: int = 1
    nutrisi: Optional[str] = None

class RecipeCreate(BaseModel):
    judul: str
    steps: Optional[str] = None
    url: Optional[str] = None
    loves: int = 0
    jumlah_resep: int = 1
    nutrisi: Optional[str] = None
    items: List[IngredientCreate]
# ================= FAVORITES =================
class FavoriteOut(BaseModel):
    id_favorit_resep: int
    id_user: int
    id_resep_makanan: int

    class Config:
        from_attributes = True


# ================= CHATBOT =================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
