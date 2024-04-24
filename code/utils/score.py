import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from torchvision.transforms import functional as TF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inception_model = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
inception_model.eval()
up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(DEVICE)

def get_feature_vectors(x, resize=False):
    if resize:
        x = up(x)
    x = inception_model(x)
    return F.softmax(x, -1).data.cpu().numpy()
#
def calculate_inception_score(features, splits=10):
    split_scores = []
    N = len(features)
    for i in range(splits):
        part = features[i * (N // splits): (i + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for pyx in part:
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores)