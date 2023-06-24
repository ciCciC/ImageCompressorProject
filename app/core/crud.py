from sqlalchemy.orm import Session
from app.core import schemas, models


def get_image(db: Session, img_id: int):
    return db.query(models.Image).filter(models.Image.id == img_id).first()


def get_images(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Image).offset(skip).limit(limit).all()


def create_image(db: Session, image: schemas.ImageCreate):
    db_image = models.Image(title=image.title, code_img=image.code_img)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image
