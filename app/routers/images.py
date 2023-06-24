from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core import schemas, crud
from app.core.dependencies import get_db

router = APIRouter(tags=["images"], prefix="/images")


@router.get("/", response_model=list[schemas.Image])
async def get_images(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    images = crud.get_images(db, skip=skip, limit=limit)
    return images


@router.get("/{id}", response_model=schemas.Image)
async def get_image(id: int, db: Session = Depends(get_db)):
    db_image = crud.get_image(db, img_id=id)

    if db_image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return db_image


@router.post("/", response_model=schemas.Image)
async def create_image(image: schemas.ImageCreate, db: Session = Depends(get_db)):
    return crud.create_image(db, image)
