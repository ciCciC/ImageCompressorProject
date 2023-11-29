import streamlit as st
from app.models import ImageCompressor
import requests
from timeit import default_timer as timer

image_compressor = ImageCompressor()
image_compressor.load_model()

BASE_URL = "http://localhost:8000"


st.write("""
    # Neural App
    """)


def inference(prompt: str):
    response: requests.Response = requests.get(f'{BASE_URL}/inference?prompt="{prompt}"')
    body = response.json()
    latents = body['latents']

    reconstruction = image_compressor.decompress(latents)
    depiction = image_compressor.depict_latents(latents)

    return reconstruction, depiction, response.elapsed


def get_latents(idx: int):
    response: requests.Response = requests.get(f'{BASE_URL}/latents/{idx}')
    body = response.json()[0]
    latents = body['payload']['latents']
    reconstruction = image_compressor.decompress(latents)
    depiction = image_compressor.depict_latents(latents)

    return reconstruction, depiction, response.elapsed


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
        start = timer()
        image, depiction, duration = get_latents(latent_idx)
        st.image([image, depiction], caption=[f'Reconstruction: {image.size}', f'Latents: {depiction.size}'],
                 clamp=True)

        st.write(f"Elapsed time: {duration.microseconds}μs")
        st.write(f'Overall time: {timer() - start}s')


if __name__ == '__main__':
    web_app()
