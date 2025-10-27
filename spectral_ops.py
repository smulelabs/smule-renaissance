import torch

class STFT:
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def __call__(self, y):
        self.window = self.window.to(y.device)
        stft_matrix = torch.stft(
            y,
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, return_complex=False, center=True, pad_mode='reflect'
        )
        return stft_matrix

class iSTFT:
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def __call__(self, X):
        self.window = self.window.to(X.device)
        X = torch.view_as_complex(X.contiguous())
        return torch.istft(
            X,
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True
        )