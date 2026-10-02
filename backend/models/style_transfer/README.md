# Neural Style Transfer

## Overview

This module creates a stylized image by combining:

- A content image, which keeps the main objects and layout
- A style image, which provides colours, textures, and artistic patterns

It uses a pretrained VGG19 CNN. The network stays frozen; only the generated image is optimized.

## Method

1. Load and resize the content and style images.
2. Extract CNN features using pretrained VGG19.
3. Calculate content loss at layer `conv4_2`.
4. Calculate style loss using Gram matrices from multiple VGG19 layers.
5. Optimize the generated image using the combined loss.
6. Save the final stylized image.

## Project structure

```ext
style_transfer/
├── api.py
├── style_transfer.py
├── requirements.txt
├── README.md
├── assets/
│   ├── content/content.jpg
│   └── style/original_abstract_style.png
└── results/
    └── content_abstract_demo.jpg
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run style transfer

```bash
python style_transfer.py \
  assets/content/content.jpg \
  assets/style/original_abstract_style.png \
  --steps 300 \
  --size 384 \
  --output results/content_abstract_demo.jpg
```

## API

Start the separate API server:

```bash
uvicorn api:app --reload --port 8001
```

Open its interactive documentation:

```text
http://127.0.0.1:8001/docs
```

### POST `/api/style-transfer`

Upload two form-data files:

- `content_image`: JPG or PNG content image
- `style_image`: JPG or PNG style image

The API returns:

```json
{
  "status": "success",
  "stylized_image": "data:image/jpeg;base64,..."
}
```
