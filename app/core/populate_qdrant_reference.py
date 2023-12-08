from settings import QDRANT_URL, QDRANT_API_KEY, DATA_DIR
from qdrant_client import QdrantClient, models
from app.models.compressor import ImageCompressor
from PIL import Image
from tqdm import tqdm
import glob
from typing import List, Tuple
import re

COLLECTION_NAME = 'reference-latents'


def get_data(dir_name: str) -> List[Tuple[Image.Image, str]]:
    return [(Image.open(img_path), img_path) for img_path in glob.glob(DATA_DIR + f'/{dir_name}/*')]


def mean_latents(compressor: ImageCompressor, image: Image.Image) -> Tuple[List, List]:
    latents_tensor, tensor_size = compressor.compress([image])
    mu = latents_tensor[0].mean(dim=1).flatten().numpy(force=True)
    return mu.tolist()


def upload_vectors(client: QdrantClient, compressor: ImageCompressor, images: List[Tuple[Image.Image, str]]):
    records = []

    for idx, image in enumerate(tqdm(images)):
        ref = re.findall(r'\d+', image[1])[0]
        # NOTE: ref should reference the related file on a pre-existing file storage
        ref = f'{ref}.encoded.png'

        mu = mean_latents(compressor, image[0])
        records.append(
            models.Record(
                id=idx,
                vector=mu,
                payload={
                    'filename': ref
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
            size=compressor.get_latent_mu_size(),
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
