# ⚡ Image to Latent Representations

<p align='center'>
  <img width='50%' src="/asset/angel.png">
</p>

[EXPERIMENTAL]

In a world of compression without storing original images, latent space representations are all you need....?

**Stack**

- Models
    - VQVAE: pretraining (see notebooks)
    - VAE Tiny: madebyollin/taesd
    - Stable Diffusion: Lykon/dreamshaper-7
- Optimization [Stable Diffusion]
    - Attention Slicing
    - LCM LoRa: latent-consistency/lcm-lora-sdv1-5
    - Fuse LoRa
- Flavour
    - 8bit latent space

### Hugging Face Spaces 🤗
- 🖼 [Image Encoder](https://huggingface.co/spaces/ciCic/ImageEncoder)
- 🔢 [Latents Decoder](https://huggingface.co/spaces/ciCic/latents-decoder)

### Preview

<p align='center'>
  <img width='50%' src="/asset/astronaut.png">
</p>

<p align='center'>
  <img width='50%' src="/asset/similar_latents.png">
</p>

### Evaluation

| Memory    | n(X) | Q1 MB  | Q2 MB  | Q3  MB | Σ  MB  |
|-----------|------|--------|--------|--------|--------|
| Originals | 99   | 0.299  | 0.338  | 0.376  | 33.631 |
| Latents   | 99   | 0.0127 | 0.0131 | 0.0134 | 1.294  |

| Elapsed time (ms)                            | µ      | m      | σ     | min    | max     | 1 run    |
|----------------------------------------------|--------|--------|-------|--------|---------|----------|
| Originals, file store (1)                    | 1.856  | 1.761  | 0.889 | 1.037  | 8.913   | -        |
| Originals, file store (99)                   | -      | -      | -     | -      | -       | 195.232  |
| Latents, file store (1)                      | 1.255  | 1.125  | 0.418 | 0.99   | 3.961   | -        |
| Latents with reconstruction, file store (1)  | -      | -      | -     | -      | -       | 40.559   |
| Latents with reconstruction, file store (99) | -      | -      | -     | -      | -       | 2522.841 |
| Latents, vector database (1)                 | 63.007 | 63.717 | 8.502 | 42.864 | 104.509 | -        |
| Latents, vector database (99)                | -      | -      | -     | -      | -       | 4189     |

## 🚀 Prerequisite

- install [miniforge](https://github.com/conda-forge/miniforge)
- create virtual env || conda
- initialize [Qdrant](https://qdrant.tech)
- from root enter the following command line

```commandline
pip install -r requirements.txt
```

```commandline
pip install python-dotenv
```

### **WINDOWS** for CUDA Deep Neural Network

- tensorflow

```commandline 
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

```commandline
pip install tensorflow==2.10
```

- [pytorch](https://pytorch.org/get-started)

```commandline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### **MACOS** for MPS

- tensorflow [installer](https://developer.apple.com/metal/tensorflow-plugin/) for MPS

```commandline 
conda install -c apple tensorflow-deps
```

```commandline
pip install tensorflow-macos==2.10.0 tensorflow-metal==0.6.0
```

- [pytorch](https://pytorch.org/get-started)

```commandline 
pip install torch torchvision
```

- app

```commandline
python app/main.py
```

- webapp

```commandline
streamlit run webapp.py
```

## 📖 DOCS

- http://127.0.0.1:8000/docs

## ARCHITECTURE

# !!Credits

- [madebyollin](https://github.com/madebyollin)
- 🤗 [Hugging Face](https://github.com/huggingface)