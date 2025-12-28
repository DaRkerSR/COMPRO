from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from database import SessionLocal
from schemas import UserCreate, UserLogin, UserResponse
import crud
import models

# Router uses '/auth' prefix; main app includes it under '/api'
router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check for existing username or email and return clear error messages
    existing = db.query(models.User).filter(
        or_(models.User.username == user.username, models.User.email == user.email)
    ).first()
    if existing:
        if existing.username == user.username:
            raise HTTPException(status_code=400, detail="Username sudah terpakai")
        if existing.email == user.email:
            raise HTTPException(status_code=400, detail="Email sudah terdaftar")

    created_user = crud.create_user(db, user)
    return UserResponse(
        id_user=created_user.id_user,
        username=created_user.username,
        email=created_user.email,
        role=created_user.role
    )


@router.post("/login", response_model=UserResponse)
def login(user: UserLogin, db: Session = Depends(get_db)):
    auth_user = crud.authenticate_user(db, user.username, user.password)
    if not auth_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return UserResponse(
        id_user=auth_user.id_user,
        username=auth_user.username,
        email=auth_user.email,
        role=auth_user.role
    )