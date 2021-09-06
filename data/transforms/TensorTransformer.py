import torch

class TensorTransformer:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)