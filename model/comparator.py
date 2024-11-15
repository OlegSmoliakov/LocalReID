import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor

log = logging.getLogger(__name__)


class Comparator:
    def __init__(self):
        self.model = FeatureExtractor(
            model_name="osnet_x0_25",  # or 'resnet50' for ResNet-50
            model_path="model/osnet_x0_25_imagenet.pth",  # pretrained Re-ID model
            device="cpu",  # or 'cpu' based on your hardware
        )

    def _get_embeddings(self, image_paths: list[str] | list[np.ndarray]):
        embeddings = self.model(image_paths)

        return embeddings

    def compare_by_similarity(
        self, img_1: str | np.ndarray, img_2: str | np.ndarray, threshold=0.7
    ):
        """if return more than 0.7 it's likely the same images"""
        embeddings = self._get_embeddings([img_1, img_2])
        embed_1, embed_2 = embeddings[0], embeddings[1]
        similarity = F.cosine_similarity(embed_1.unsqueeze(0), embed_2.unsqueeze(0)).item()
        log.debug(f"Similarity: {similarity:.4f}")

        return True if similarity >= threshold else False

    def compare_by_distance(self, img_1: str | np.ndarray, img_2: str | np.ndarray, threshold=2):
        """if return 1 or 2 it's likely the same images"""
        embeddings = self._get_embeddings([img_1, img_2])
        embed_1, embed_2 = embeddings[0], embeddings[1]
        distance = torch.norm(embed_1 - embed_2, p=2).item()
        log.debug(f"Distance: {distance:.4f}")

        return True if distance <= threshold else False


if __name__ == "__main__":
    image_1 = "image_1.png"
    image_2 = "image_2.png"

    import cv2

    image_1 = cv2.imread(image_1)
    image_2 = cv2.imread(image_2)

    comparator = Comparator()
    sim = comparator.compare_by_similarity(image_1, image_2)
    print(sim)
