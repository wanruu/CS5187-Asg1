import torch
from utils import DEVICE
import torchvision.models as models


def vgg11_feat(imgs):
    model = models.vgg11(pretrained=True).to(DEVICE)
    feat_extractor = model.features
    feat_extractor.eval()
    features = feat_extractor(imgs)
    features_np = features.cpu().detach().numpy()
    return features_np  # [B, C, H, W]


if __name__ == "__main__":
    pass

