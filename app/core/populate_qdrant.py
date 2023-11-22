from settings import QDRANT_URL, COLLECTION_NAME, QDRANT_API_KEY, DATA_DIR
from qdrant_client import QdrantClient, models
from app.models.compressor import ImageCompressor
from PIL import Image
from tqdm import tqdm
import glob
from typing import List


def get_data(dir_name) -> List[Image.Image]:
    return [Image.open(img_path) for img_path in glob.glob(DATA_DIR + f'/{dir_name}/*')]


def upload_vectors(client: QdrantClient, compressor: ImageCompressor, images: List[Image.Image]):
    client.upload_records(
        collection_name=COLLECTION_NAME,
        records=[
            models.Record(
                id=idx,
                vector=compressor.vector_ndarray(compressor.compress([image])[0]).tolist(),
                payload={'dims': compressor.get_latent_dims()}
            )
            for idx, image in enumerate(tqdm(images))
        ],
    )


def create_collection(client: QdrantClient, compressor: ImageCompressor):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=compressor.get_latent_flat_size(),
            distance=models.Distance.COSINE,
            on_disk=True
        )
    )


def populate():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    images = get_data('dream')
    compressor = ImageCompressor()
    compressor.load_model()

    create_collection(client, compressor)
    upload_vectors(client, compressor, images)


if __name__ == '__main__':
    populate()
