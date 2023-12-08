from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from core.qdrant_service import QdrantService
from core.neural_service import NeuralService
from app.core.settings import DATA_DIR
import logging
import datetime

logging.basicConfig(format="%(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

title = 'Image Compressor API'
app = FastAPI(version='1.0.0', title=title)

app.mount('/dream', StaticFiles(directory=DATA_DIR + '/dream'), name="dream")
app.mount('/data', StaticFiles(directory=DATA_DIR + '/compressed'), name="encoded")

app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

qdrant_service = QdrantService()
neural_service = NeuralService()


class Latents(BaseModel):
    collection: str
    mu: list


@app.post("/latents/search")
async def search(latents: Latents):
    logger.info(f'search latents, {datetime.datetime.now()}')
    return await qdrant_service.search(latents.collection, latents.mu)


@app.get("/latents")
async def get_latents(start: int, end: int):
    logger.info(f'get_latents, {datetime.datetime.now()}')
    return await qdrant_service.get_latents(start, end)


@app.get("/latents/{idx}")
async def get_latents(idx: int):
    logger.info(f'get_latents idx, {datetime.datetime.now()}')
    return await qdrant_service.get_latents(idx)


@app.get("/inference")
async def inference(prompt: str):
    latents, shape, is_nsfw = await neural_service.prompt_inference_async(prompt)
    return {
        'latents': latents.tolist(),
        'is_nsfw': is_nsfw,
        'shape': shape
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(f"{__name__}:app", host="0.0.0.0", port=8000, log_level="info")
