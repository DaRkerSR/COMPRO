from sqlalchemy import Column, Integer, String, ForeignKey, Float, Text, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime 


class User(Base):
    __tablename__ = "users"

    id_user = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user")

    favorites = relationship("FavoriteResep", back_populates="user")


class BahanMakan(Base):
    __tablename__ = "bahan_makan"

    id_bahan = Column(Integer, primary_key=True, index=True)
    nama_bahan = Column(String(100), unique=True, index=True)
    berat = Column(Float)
    jumlah = Column(Integer)


class ResepMakanan(Base):
    __tablename__ = "resep_makanan"

    id_resep_makanan = Column(Integer, primary_key=True, index=True)
    
      # Kolom baru (TAMBAHAN)
    judul = Column(String(255), nullable=False, index=True)
    steps = Column(Text)  # cara memasak
    url = Column(String(500))  # link original (opsional)
    loves = Column(Integer, default=0)  # rating/jumlah sukai
    created_at = Column(DateTime, default=datetime.utcnow)

    jumlah_resep = Column(Integer)
    nutrisi = Column(String(255))

    detail = relationship(
        "DetailResep",
        back_populates="resep",
        cascade="all, delete-orphan",
    )


class DetailResep(Base):
    __tablename__ = "detail_resep"

    id_detail_resep = Column(Integer, primary_key=True, index=True)
    id_resep_makanan = Column(
        Integer, ForeignKey("resep_makanan.id_resep_makanan")
    )
    id_bahan = Column(Integer, ForeignKey("bahan_makan.id_bahan"), nullable=True)

    bahan_digunakan = Column(Text)
    jumlah = Column(Integer)
    nutrisi = Column(String(255))

    resep = relationship("ResepMakanan", back_populates="detail")
    bahan = relationship("BahanMakan")


class FavoriteResep(Base):
    __tablename__ = "favorite_resep"

    id_favorit_resep = Column(Integer, primary_key=True, index=True)
    id_user = Column(Integer, ForeignKey("users.id_user"))
    id_resep_makanan = Column(Integer, ForeignKey("resep_makanan.id_resep_makanan"))

    user = relationship("User", back_populates="favorites")
    resep = relationship("ResepMakanan")
