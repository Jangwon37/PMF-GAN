import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from scipy.stats import entropy
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from torchvision.transforms import functional as TF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index, :, :, :]

    def __len__(self):
        return (self.orig).shape[0]

def inception_score(imgs, cuda=True, resize=False, splits=1, batch_size=64) -> Tuple[float, float]:
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(DEVICE)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, -1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(torch.utils.data.DataLoader(imgs, batch_size=batch_size)):
        batch = batch.to(DEVICE)
        batch_size_i = batch.shape[0]

        # Ensure the batch is correctly sized
        assert batch.shape == (batch_size_i, 3, imgs.orig.shape[2], imgs.orig.shape[3]), "Batch shape mismatch"

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def get_feature_vector(img, model) -> np.ndarray:
    if not isinstance(img, torch.Tensor):
        img = TF.to_tensor(img)

    img = TF.resize(img, (299, 299))
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().cpu().numpy()

