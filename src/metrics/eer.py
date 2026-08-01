import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


def compute_eer(scores, labels):

    thresholds = np.linspace(
        0,
        1,
        10000
    )

    fars = []
    frrs = []

    for threshold in thresholds:

        predict = scores >= threshold

        false_accept = (
            (predict == 1)
            &
            (labels == 0)
        ).sum()

        false_reject = (
            (predict == 0)
            &
            (labels == 1)
        ).sum()

        bona = (labels == 0).sum()
        spoof = (labels == 1).sum()

        far = false_accept / max(bona, 1)
        frr = false_reject / max(spoof, 1)

        fars.append(far)
        frrs.append(frr)


    fars = np.array(fars)
    frrs = np.array(frrs)

    idx = np.argmin(
        np.abs(fars - frrs)
    )

    return float(
        (fars[idx] + frrs[idx]) / 2
    )


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