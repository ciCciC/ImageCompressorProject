from pydantic import BaseModel


# Base model
class ImageBase(BaseModel):
    title: str
    code_img: str


# When CREATING
class ImageCreate(ImageBase):
    pass


# When READING
class Image(ImageBase):
    id: int

    class Config:
        orm_mode = True
