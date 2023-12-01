from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from core.qdrant_service import QdrantService
from core.neural_service import NeuralService

title = 'Image Compressor API'
app = FastAPI(version='1.0.0', title=title)

app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

qdrant_service = QdrantService()
neural_service = NeuralService()


class Latents(BaseModel):
    mu: list


@app.post("/latents/search")
async def search(latents: Latents):
    return await qdrant_service.search(latents.mu)


@app.get("/latents")
async def get_latents():
    return await qdrant_service.get_all_latents()


@app.get("/latents/{idx}")
async def get_latents(idx: int):
    return await qdrant_service.get_latents(idx)


@app.get("/inference")
async def inference(prompt: str):
    latents, shape, is_nsfw = await neural_service.prompt_inference_async(prompt)
    return {
        'latents': latents.tolist(),
        'is_nsfw': is_nsfw,
        'shape': shape
    }


@app.get("/")
async def health_check():
    return {"msg": f"{title} is up and running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(f"{__name__}:app", host="0.0.0.0", port=8000, log_level="info")
