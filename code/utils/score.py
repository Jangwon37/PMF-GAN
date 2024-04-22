import numpy as np
import torch
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from torchvision.transforms import functional as TF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inception_model = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
inception_model.eval()

def preprocess_images(imgs):
    imgs = TF.resize(imgs, (299, 299))
    imgs = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return imgs.to(DEVICE)
#
def get_feature_vectors(imgs):
    imgs = preprocess_images(imgs)
    with torch.no_grad():
        outputs = inception_model(imgs)
    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
    return F.softmax(logits, dim=1).cpu().numpy()
#
def calculate_inception_score(features, splits=1):
    scores = []
    N = len(features)
    for i in range(splits):
        part = features[i * (N // splits): (i + 1) * (N // splits)]
        py = np.mean(part, axis=0)
        for pyx in part:
            scores.append(entropy(pyx, py))
    is_score = np.exp(np.mean(scores))
    return is_score