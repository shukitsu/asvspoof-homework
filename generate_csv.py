import torch
import pandas as pd
import hydra
from hydra.utils import instantiate
from tqdm.auto import tqdm
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed

@hydra.main(version_base=None, config_path="src/configs", config_name="generate_csv")
def main(config):
    set_random_seed(config.inferencer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloaders, _ = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    checkpoint_path = "saved/testing/model_best.pth"
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    all_scores = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(dataloaders["test"], desc="Inference"):
            specs = batch["data_object"].to(device)
            logits = model(specs)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            filenames = batch["filename"]
            all_scores.extend(probs)
            all_filenames.extend(filenames)

    df = pd.DataFrame({"filename": all_filenames, "score": all_scores})
    df.to_csv("your_university_email.csv", index=False)
    print("CSV saved!")

if __name__ == "__main__":
    main()