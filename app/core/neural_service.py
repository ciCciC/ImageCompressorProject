from app.core.settings import QDRANT_URL, COLLECTION_NAME, QDRANT_API_KEY
from qdrant_client import AsyncQdrantClient, models
from app.models.compressor import ImageCompressor


class NeuralService:
    def __init__(self):
        self.compressor = None
        self.qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
        self.prepare_compressor()

    def prepare_compressor(self):
        self.compressor = ImageCompressor()
        self.compressor.load_model()

    async def get_all_latents(self):
        collection: models.CollectionInfo = await self.qdrant_client.get_collection(COLLECTION_NAME)
        return await self.qdrant_client.retrieve(COLLECTION_NAME, ids=list(range(collection.vectors_count)),
                                                 with_vectors=True)

    async def get_latent(self, idx: int):
        return await self.qdrant_client.retrieve(COLLECTION_NAME, ids=[idx], with_vectors=True)
