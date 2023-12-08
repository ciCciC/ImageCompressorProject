from qdrant_client import AsyncQdrantClient
from app.core.settings import QDRANT_URL, COLLECTION_NAME, QDRANT_API_KEY


class QdrantService:

    def __init__(self):
        self.qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)

    async def get_latents(self, start: int, end: int = None):
        ids = list(range(start, end)) if end else [start]
        return await self.qdrant_client.retrieve(COLLECTION_NAME, ids=ids, with_vectors=True)

    async def search(self, collection, mu, limit=5):
        return await self.qdrant_client.search(
            collection_name=collection,
            query_vector=mu,
            limit=limit
        )
