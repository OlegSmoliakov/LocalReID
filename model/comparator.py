import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor

log = logging.getLogger(__name__)


class Comparator:
    def __init__(self):
        self.model = FeatureExtractor(
            model_name="osnet_x1_0",  # or 'resnet50' for ResNet-50
            model_path="model/osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
            # model_path="model/osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
            # model_path="model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
            # model_path="model/osnet_x1_0_imagenet.pt",  # pretrained Re-ID model
            # model_name="osnet_x0_25",  # or 'resnet50' for ResNet-50
            # model_path="model/osnet_x0_25_imagenet.pth",  # pretrained Re-ID model
            device="cpu",  # or 'cpu' based on your hardware
        )

    def _get_embeddings(self, imgs: list[int, np.ndarray]):
        embeddings = self.model(imgs)

        return embeddings

    def get_similarity_map(self, probe: np.ndarray, gallery: list[np.ndarray], threshold=0.7):
        similarity_map: dict[int, float] = {}
        gallery_and_probe = self._preprocess(probe, gallery)
        embeddings = self._get_embeddings(gallery_and_probe)
        probe_embed, gallery_embed = embeddings[-1], embeddings[:-1]
        for i, embed in enumerate(gallery_embed):
            if similarity := self.compare_by_similarity(probe_embed, embed, threshold):
                similarity_map[i] = similarity

        # sort desc by value
        similarity_map = dict(
            sorted(similarity_map.items(), key=lambda item: item[1], reverse=True)
        )

        return similarity_map

    def compare_by_similarity(self, embed_1, embed_2, threshold=0.7):
        """if return more than 0.7 it's likely the same images"""

        similarity = F.cosine_similarity(embed_1.unsqueeze(0), embed_2.unsqueeze(0)).item()
        log.debug(f"Similarity: {similarity:.4f}")

        return similarity if similarity >= threshold else 0

    def _preprocess(self, probe: np.ndarray, gallery: list[np.ndarray]):
        images = [*gallery, probe]
        return images
        # images = [*gallery]

        # Find the smallest image dimensions
        min_height, min_width = min((img.shape[:2] for img in images), key=lambda x: x[0] * x[1])
        log.debug(f"Min height: {min_height}, Min width: {min_width}")
        gallery_and_probe = [self._resize_images(img, min_width, min_height) for img in images]
        # gallery_and_probe.append(probe)

        return gallery_and_probe

    def _resize_images(self, img, min_width, min_height):
        h, w = img.shape[:2]
        scale = min(min_width / w, min_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        log.debug(f"New width: {new_w}, New height: {new_h}")
        return cv2.resize(img, (new_w, new_h))

    def compare_by_distance(self, probe: str | np.ndarray, img_2: str | np.ndarray, threshold=2):
        """if return 1 or 2 it's likely the same images"""
        embeddings = self._get_embeddings([probe, img_2])
        embed_1, embed_2 = embeddings[0], embeddings[1]
        distance = torch.norm(embed_1 - embed_2, p=2).item()
        log.debug(f"Distance: {distance:.4f}")

        return True if distance <= threshold else False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(levelname)s: %(message)s")

    image_0 = "cache/active_person_0.png"
    image_1 = "cache/active_person_1.png"
    # image_2 = "cache/new_person_2.png"
    image_2 = "cache/active_person_2.png"
    image_second_cam = "cache/second_cam_person_2.png"

    gallery = map(cv2.imread, [image_0, image_1, image_2])
    image_second_cam = cv2.imread(image_second_cam)

    comparator = Comparator()

    sim_map = comparator.get_similarity_map(image_second_cam, gallery)
    print(sim_map)

# downsize to small one include probe
# 2024-11-18 17:46:03,872: DEBUG: Similarity: 0.6447 /2
# 2024-11-18 17:46:03,872: DEBUG: Similarity: 0.6804 /1
# 2024-11-18 17:46:03,872: DEBUG: Similarity: 0.6401 /3

# downsize to small exclude probe
# 2024-11-18 17:50:32,738: DEBUG: Similarity: 0.5343 /2
# 2024-11-18 17:50:32,738: DEBUG: Similarity: 0.6568 /1
# 2024-11-18 17:50:32,738: DEBUG: Similarity: 0.5275 /3


# original size ImageNet
# 2024-11-18 17:46:56,918: DEBUG: Similarity: 0.6654 /2
# 2024-11-18 17:46:56,918: DEBUG: Similarity: 0.7136 /1
# 2024-11-18 17:46:56,919: DEBUG: Similarity: 0.5275 /3


# original size Market
# 2024-11-18 18:07:41,131: DEBUG: Similarity: 0.5312 /3
# 2024-11-18 18:07:41,131: DEBUG: Similarity: 0.6458 /1
# 2024-11-18 18:07:41,131: DEBUG: Similarity: 0.5916 /2


# original size msmt17
# 2024-11-18 18:12:23,312: DEBUG: Similarity: 0.5090 /3
# 2024-11-18 18:12:23,312: DEBUG: Similarity: 0.6123 /2
# 2024-11-18 18:12:23,312: DEBUG: Similarity: 0.6735 /1 !

# original size Duke
# 2024-11-18 18:14:03,456: DEBUG: Similarity: 0.5528
# 2024-11-18 18:14:03,457: DEBUG: Similarity: 0.5907
# 2024-11-18 18:14:03,457: DEBUG: Similarity: 0.8634
