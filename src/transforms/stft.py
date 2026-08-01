import torch
from torch import nn


class STFTTransform(nn.Module):
    """
    Converts waveform to log-magnitude STFT.
    Output shape:
        [1, freq_bins, time_frames]
    """

    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer(
            "window",
            torch.hann_window(win_length)
        )

    def forward(self, waveform):
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        spec = torch.abs(spec)
        spec = torch.log(spec + 1e-6)
        spec = spec.unsqueeze(0)

        return spec