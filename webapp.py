import streamlit as st
from app.models import ImageCompressor
from torchvision.transforms.functional import to_pil_image
import requests
from timeit import default_timer as timer
from typing import List

image_compressor = ImageCompressor()
image_compressor.load_model()

BASE_URL = "http://localhost:8000"

st.write("""
    # Compression towards Latent Space
    """)


def inference(prompt: str):
    response: requests.Response = requests.get(f'{BASE_URL}/inference?prompt="{prompt}"')
    body = response.json()
    latents = body['latents']

    reconstruction = image_compressor.decompress(latents)
    depiction = image_compressor.depict_latents(latents)

    return reconstruction, depiction, response.elapsed


def get_reconstructed_latents(idx: int):
    response: requests.Response = requests.get(f'{BASE_URL}/latents/{idx}')
    body = response.json()[0]
    latents = body['payload']['latents']
    vector = body['vector']
    reconstruction = image_compressor.decompress(latents)
    depiction = image_compressor.depict_latents(latents)

    return vector, reconstruction, depiction, response.elapsed


def search(latents: List):
    response: requests.Response = requests.post(f'{BASE_URL}/latents/search', json={'mu': latents})
    latent_space_block = [vector["payload"]['latents'] for vector in response.json()]
    scores = [vector["score"] for vector in response.json()]
    reconstructed_latents = image_compressor.decompress_batch(latent_space_block)
    reconstructed_images = [to_pil_image(tensor) for tensor in reconstructed_latents]
    return reconstructed_images, scores


def web_app_prompting():
    prompt = st.text_input('Prompt')

    if prompt:
        start = timer()
        image, depiction, duration = inference(prompt)
        st.write(f"Elapsed time: {duration}")
        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)

        st.write(f"Elapsed time: {duration.microseconds}μs")
        st.write(f'Overall time: {timer() - start}s')


def web_app():
    latent_idx = st.text_input('Latent idx')

    if latent_idx and latent_idx.isdigit():
        o_s = timer()
        r_s = timer()
        vector, image, depiction, duration = get_reconstructed_latents(latent_idx)
        s_r_s = timer()
        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)
        s_o_s = timer()

        st.write(f"Elapsed time: {duration.microseconds / 1e3}ms")
        st.write(f"Reconstruction time: {s_r_s - r_s}s")
        st.write(f'Overall time: {s_o_s - o_s}s')


def web_app_similar():
    latent_idx = st.text_input('Latent idx')

    if latent_idx and latent_idx.isdigit():
        vector, image, depiction, duration = get_reconstructed_latents(latent_idx)

        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)

        state = st.button('Similar latents')
        if state:
            r_s = timer()
            top_k, scores = search(vector)
            s_r_s = timer()

            grid_size = len(top_k)
            grid = st.columns(grid_size)
            for idx in range(grid_size):
                with grid[idx]:
                    reconstructed = top_k[idx]
                    score = scores[idx]
                    st.image(reconstructed, caption=f'Score: {score}')

            st.write(f"Elapsed time: {duration.microseconds / 1e3}ms")
            st.write(f"Reconstruction time: {s_r_s - r_s}s")
            st.write(f"Reconstruction dimensions: 512x512")


if __name__ == '__main__':
    # web_app_similar()
    web_app()
