from settings import QDRANT_URL, COLLECTION_NAME, QDRANT_API_KEY, DATA_DIR
from qdrant_client import QdrantClient, models
from app.models.compressor import ImageCompressor
from PIL import Image
from tqdm import tqdm
import glob
from typing import List, Tuple


def get_paths(dir_name: str) -> List[str]:
    return [img_path for img_path in glob.glob(DATA_DIR + f'/{dir_name}/*')]


def get_data(paths: List[str]) -> List[Image.Image]:
    return [Image.open(img_path) for img_path in paths]


def to_image_latent(compressor: ImageCompressor, image: Image.Image) -> Tuple[List, List]:
    latents_tensor, tensor_size = compressor.compress([image])
    mu = latents_tensor[0].mean(dim=1).flatten().numpy(force=True)
    vectorized = compressor.vector_ndarray(latents_tensor[0])
    return mu.tolist(), vectorized.tolist()


def upload_vectors(client: QdrantClient, compressor: ImageCompressor, images: List[Image.Image], paths: List[str]):
    records = []

    for idx, image in enumerate(tqdm(images)):
        mu, vectorized = to_image_latent(compressor, image)
        records.append(
            models.Record(
                id=idx,
                vector=mu,
                payload={
                    'name': paths[idx],
                    'latents': vectorized
                }
            )
        )

    client.upload_records(
        collection_name=COLLECTION_NAME,
        records=records,
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

    paths = get_paths('dream')
    images = get_data(paths)
    compressor = ImageCompressor()
    compressor.load_model()

    create_collection(client, compressor)
    upload_vectors(client, compressor, images, paths)


if __name__ == '__main__':
    populate()
