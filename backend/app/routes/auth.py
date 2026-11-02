import datetime
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, Response
from pydantic import BaseModel, EmailStr, Field
from app.database import users_collection
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    JWT_EXPIRATION_HOURS,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    role: Optional[str] = "customer"
    preferred_categories: Optional[List[str]] = []


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    preferred_categories: List[str] = []
    created_at: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, response: Response):
    existing = users_collection.find_one({"email": req.email.lower()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists"
        )
    
    user_id = str(uuid.uuid4())
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    role = "admin" if req.email.lower().startswith("admin@") else (req.role or "customer")

    new_user = {
        "id": user_id,
        "name": req.name.strip(),
        "email": req.email.lower().strip(),
        "password_hash": hash_password(req.password),
        "role": role,
        "preferred_categories": req.preferred_categories or [],
        "created_at": now_iso,
        "updated_at": now_iso
    }

    users_collection.insert_one(new_user)
    
    # Generate token
    token = create_access_token({"sub": new_user["email"], "uid": user_id, "role": role})
    
    # Set HTTP-only cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=JWT_EXPIRATION_HOURS * 3600
    )

    return {
        "message": "User registered successfully",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": new_user["name"],
            "email": new_user["email"],
            "role": role,
            "preferred_categories": new_user["preferred_categories"],
            "created_at": now_iso
        }
    }


@router.post("/login")
def login(req: LoginRequest, response: Response):
    user = users_collection.find_one({"email": req.email.lower().strip()})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    token = create_access_token({"sub": user["email"], "uid": user.get("id"), "role": user.get("role", "customer")})
    
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=JWT_EXPIRATION_HOURS * 3600
    )

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.get("id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "role": user.get("role", "customer"),
            "preferred_categories": user.get("preferred_categories", []),
            "created_at": user.get("created_at")
        }
    }


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}


@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user.get("id"),
            "name": current_user.get("name"),
            "email": current_user.get("email"),
            "role": current_user.get("role", "customer"),
            "preferred_categories": current_user.get("preferred_categories", []),
            "created_at": current_user.get("created_at")
        }
    }

