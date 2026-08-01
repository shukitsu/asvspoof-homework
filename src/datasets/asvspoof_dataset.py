from pathlib import Path
import torch
import librosa
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
                label = 0 if parts[-1] == "bonafide" else 1
                path = self.dataset_dir / "flac" / f"{filename}.flac"
                index.append({
                    "path": str(path),
                    "label": label,
                    "filename": filename,
                })
        return index

    def load_object(self, path):
        waveform, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return torch.tensor(waveform, dtype=torch.float32)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["label"]
        data_filename = data_dict["filename"]

        instance_data = {
            "data_object": data_object,
            "labels": data_label,
            "filename": data_filename,
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data