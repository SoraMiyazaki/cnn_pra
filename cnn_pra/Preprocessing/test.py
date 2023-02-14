import numpy as np
import torch
from PIL import Image

t = torch.Tensor([
    [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
])

hoge = t.argmax(0)
hoge = (hoge*255).to(torch.uint8)
image = Image.fromarray(hoge.to("cpu").detach().numpy().copy())
image.save("hoge.jpg", quality=100)
