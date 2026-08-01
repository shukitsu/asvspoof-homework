import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


def compute_eer(scores, labels):

    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    fars = []
    frrs = []

    for threshold in thresholds:

        predict = scores >= threshold

        false_accept = np.sum(
            (predict == 1) & (labels == 0)
        )

        false_reject = np.sum(
            (predict == 0) & (labels == 1)
        )

        bona = np.sum(labels == 0)
        spoof = np.sum(labels == 1)

        far = false_accept / max(bona, 1)
        frr = false_reject / max(spoof, 1)

        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    idx = np.argmin(np.abs(fars - frrs))

    return (fars[idx] + frrs[idx]) / 2


class EER(BaseMetric):

    def __call__(
        self,
        logits,
        labels,
        **kwargs
    ):

        scores = torch.softmax(
            logits,
            dim=-1
        )[:, 1]

        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        return compute_eer(
            scores,
            labels
        )