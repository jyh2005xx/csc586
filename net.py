import torch
from utils.torch_utils import gaussian, circle_mask, half_mask
import torchvision.models as models

class BackBoneVGG(torch.nn.Module):
    def __init__(self, config):
        super(BackBoneVGG, self).__init__()

        vgg16_feature = list(models.vgg16(pretrained=True).features.children())
        # not using the last maxpool layer
        self.features = torch.nn.Sequential(*vgg16_feature[:-1])
        # Fix the layers before conv3:
        # for layer in range(10):
        for layer in range(len(self.features)):
            for p in self.features[layer].parameters(): p.requires_grad = config.tune_vgg

    def forward(self, image):
        featuremap = self.features(image)
        return featuremap