import torch
from hydra import initialize, compose
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed


def main():

    with initialize(
        version_base=None,
        config_path="src/configs",
    ):
        config = compose(
            config_name="baseline"
        )


    set_random_seed(config.trainer.seed)


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("DEVICE:", device)


    # создаём dataloader
    dataloaders, batch_transforms = get_dataloaders(
        config,
        device,
    )


    print(
        "TRAIN SIZE:",
        len(dataloaders["train"].dataset)
    )


    # берём один batch
    batch = next(
        iter(dataloaders["train"])
    )


    print(
        "DATA SHAPE:",
        batch["data_object"].shape
    )

    print(
        "LABEL SHAPE:",
        batch["labels"].shape
    )

    print(
        "LABELS:",
        batch["labels"]
    )


    # модель
    model = instantiate(
        config.model
    ).to(device)


    x = batch["data_object"].to(device)


    output = model(x)


    print(
        "LOGITS SHAPE:",
        output["logits"].shape
    )


    print("\nPipeline OK!")


if __name__ == "__main__":
    main()