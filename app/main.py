from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

title = 'ImageCompressorAPI'
app = FastAPI(title=title)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    return {"message": title}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/images", response_model=list[schemas.Image])
async def get_images(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    images = crud.get_images(db, skip=skip, limit=limit)
    return images


@app.get("/images/{id}", response_model=schemas.Image)
async def get_image(id: int, db: Session = Depends(get_db)):
    db_image = crud.get_image(db, img_id=id)

    if db_image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return db_image


@app.post("/images", response_model=schemas.Image)
async def create_image(image: schemas.ImageCreate, db: Session = Depends(get_db)):
    return crud.create_image(db, image)
