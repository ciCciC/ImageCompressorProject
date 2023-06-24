from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from app.routers import images as images_router
from app.core.database import create_db

title = 'ImageCompressorAPI'
app = FastAPI(version='1.0.0', title=title)

app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

app.include_router(images_router.router)


@app.on_event("startup")
def on_startup():
    create_db()


@app.get("/")
async def health_check():
    return {"msg": f"Welcome to {title}, the server is up and running"}


if __name__ == "__main__":
    uvicorn.run("main:app", log_level="info", reload=True)
