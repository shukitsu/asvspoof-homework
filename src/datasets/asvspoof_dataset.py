from pathlib import Path
import torch


from src.datasets.base_dataset import BaseDataset


class ASVSpoofDataset(BaseDataset):

    def __init__(
        self,
        dataset_dir,
        protocol_path,
        sample_rate=16000,
        *args,
        **kwargs,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.protocol_path = Path(protocol_path)
        self.sample_rate = sample_rate

        index = self._create_index()

        super().__init__(
            index=index,
            *args,
            **kwargs,
        )


    def _create_index(self):

        index = []

        with open(self.protocol_path) as f:
            for line in f:

                parts = line.strip().split()

                filename = parts[1]

                label = (
                    0
                    if parts[-1] == "bonafide"
                    else 1
                )

                path = (
                    self.dataset_dir
                    / "flac"
                    / f"{filename}.flac"
                )

                index.append(
                    {
                        "path": str(path),
                        "label": label,
                    }
                )

        return index

    def load_object(self, path):
        import librosa

        waveform, sr = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True,
        )

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        return waveform