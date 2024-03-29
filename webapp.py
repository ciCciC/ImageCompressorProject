import streamlit as st

from app.models import ImageCompressor, FacebookDinoV2
from torchvision.transforms.functional import to_pil_image
import requests
from timeit import default_timer as timer
from typing import List
from PIL import Image

image_compressor = ImageCompressor()
image_compressor.load_model()

dino_model = FacebookDinoV2()
dino_model.load_model()

BASE_URL = "http://localhost:8000"

st.write("""
    # Compression as Latent Space Representations
    """)


def inference(prompt: str):
    response: requests.Response = requests.get(f'{BASE_URL}/inference?prompt="{prompt}"')
    body = response.json()
    st.write(body.keys())
    latents = body['latents']

    reconstruction = image_compressor.decompress(latents)
    depiction = image_compressor.depict_latents(latents)

    return reconstruction, depiction, response.elapsed


def web_app_prompting():
    example_prompt = 'portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors'
    prompt = st.text_input('Prompt', placeholder=example_prompt)
    st.markdown(example_prompt)

    if prompt:
        start = timer()
        image, depiction, elapsed_time = inference(prompt)
        end = timer()
        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)

        st.write(f"Elapsed time (ms)", elapsed_time.total_seconds() * 1e3)
        st.write(f'Overall time (s)', end - start)


def get_reconstructed_latents(idx: int):
    response: requests.Response = requests.get(f'{BASE_URL}/latents/{idx}')
    body = response.json()[0]
    latents = body['payload']['latents']
    vector = body['vector']
    reconstruction = image_compressor.decompress(latents)
    depiction = image_compressor.depict_latents(latents)

    return vector, reconstruction, depiction, response.elapsed


def get_reconstructed_from_image(idx: int):
    response = requests.get(f"{BASE_URL}/data/{idx}.encoded.png", stream=True)
    latents_image = Image.open(response.raw)

    reconstruction = image_compressor.decompress_by_image(latents_image)

    return reconstruction, latents_image, response.elapsed


def search(latents: List):
    response: requests.Response = requests.post(f'{BASE_URL}/latents/search',
                                                json={'mu': latents, 'collection': 'latent-images'})
    latent_space_block = [vector["payload"]['latents'] for vector in response.json()]
    scores = [vector["score"] for vector in response.json()]
    reconstructed_latents = image_compressor.decompress_batch(latent_space_block)
    reconstructed_images = [to_pil_image(tensor) for tensor in reconstructed_latents]
    return reconstructed_images, scores, response.elapsed


def search_reference(latents: List):
    response: requests.Response = requests.post(f'{BASE_URL}/latents/search',
                                                json={'mu': latents, 'collection': 'reference-latents'})
    filename_block = [vector["payload"]['filename'] for vector in response.json()]
    latent_space_block = [Image.open(requests.get(f"{BASE_URL}/data/{filename}", stream=True).raw) for filename in
                          filename_block]

    scores = [vector["score"] for vector in response.json()]
    reconstructed_latents = image_compressor.decompress_batch_by_image(latent_space_block)
    reconstructed_images = [to_pil_image(tensor) for tensor in reconstructed_latents]
    return reconstructed_images, scores, response.elapsed


def web_app():
    latent_idx = st.text_input('Latent idx')

    if latent_idx and latent_idx.isdigit():
        original_image = Image.open(requests.get(f"{BASE_URL}/dream/{latent_idx}.png", stream=True).raw)

        o_s = timer()
        r_s = timer()
        vector, image, depiction, elapsed_time = get_reconstructed_latents(latent_idx)
        s_r_s = timer()
        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)
        s_o_s = timer()

        st.write("Elapsed time (ms)", elapsed_time.total_seconds() * 1e3)
        st.write("Reconstruction time (s)", s_r_s - r_s)
        st.write('Overall time (s)', s_o_s - o_s)

        embeddings = dino_model.get_embeddings([original_image, image])
        cosine_score = dino_model.compute_similarity(embeddings)

        st.write('Cosine(original, reconstruction)', cosine_score)


def web_app_similar():
    latent_idx = st.text_input('Latent idx')

    if latent_idx and latent_idx.isdigit():
        vector, image, depiction, duration = get_reconstructed_latents(latent_idx)

        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)

        state = st.button('Similar latents')
        if state:
            r_s = timer()
            top_k, scores, elapsed_time = search(vector)
            s_r_s = timer()

            grid_size = len(top_k)
            grid = st.columns(grid_size)
            for idx in range(grid_size):
                with grid[idx]:
                    reconstructed = top_k[idx]
                    score = scores[idx]
                    st.image(reconstructed, caption=f'Score: {score}')

            st.write(f"Search (elapsed) time (ms)", elapsed_time.total_seconds() * 1e3)
            st.write(f"Reconstruction time (s)", s_r_s - r_s)
            st.write(f"Reconstruction dimensions: 512x512")


def web_app_file_store():
    latent_idx = st.text_input('Latent idx')

    if latent_idx and latent_idx.isdigit():
        original_image = Image.open(requests.get(f"{BASE_URL}/dream/{latent_idx}.png", stream=True).raw)

        o_s = timer()
        r_s = timer()

        reconstruction, latents_image, elapsed_time = get_reconstructed_from_image(latent_idx)
        s_r_s = timer()

        st.image([reconstruction, latents_image],
                 caption=[f'Reconstruction: {reconstruction.size}', f'Latents: {latents_image.size}'],
                 clamp=True)
        s_o_s = timer()

        st.write("Elapsed time (ms)", elapsed_time.total_seconds() * 1e3)
        st.write("Reconstruction time (s)", s_r_s - r_s)
        st.write('Overall time (s)', s_o_s - o_s)

        embeddings = dino_model.get_embeddings([original_image, reconstruction])
        cosine_score = dino_model.compute_similarity(embeddings)

        st.write('Cosine(original, reconstruction)', cosine_score)


def web_app_file_store_similar():
    latent_idx = st.text_input('Latent idx')

    if latent_idx and latent_idx.isdigit():

        vector, image, depiction, duration = get_reconstructed_latents(latent_idx)

        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)

        state = st.button('Similar latents')
        if state:
            r_s = timer()
            top_k, scores, elapsed_time = search_reference(vector)
            s_r_s = timer()

            grid_size = len(top_k)
            grid = st.columns(grid_size)
            for idx in range(grid_size):
                with grid[idx]:
                    reconstructed = top_k[idx]
                    score = scores[idx]
                    st.image(reconstructed, caption=f'Score: {score}')

            st.write(f"Search (elapsed) time (ms)", elapsed_time.total_seconds() * 1e3)
            st.write(f"Reconstruction time (s)", s_r_s - r_s)
            st.write(f"Reconstruction dimensions: 512x512")


if __name__ == '__main__':
    # web_app_prompting()
    # web_app()
    # web_app_file_store()
    # web_app_similar()
    web_app_file_store_similar()
