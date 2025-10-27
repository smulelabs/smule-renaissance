import argparse
import torch
import torchaudio
from pathlib import Path
from spectral_ops import STFT, iSTFT
from model import Renaissance

def load_and_preprocess_audio(input_path, device, dtype):
    waveform, sr = torchaudio.load(input_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"Converted to mono from {waveform.shape[0]} channels")
    
    if sr != 48000:
        print(f"Resampling from {sr} Hz to 48000 Hz")
        resampler = torchaudio.transforms.Resample(sr, 48000)
        waveform = resampler(waveform)

    waveform = torchaudio.functional.highpass_biquad(
        waveform, 48000, cutoff_freq=60.0
    )
    
    waveform = waveform.to(device).to(dtype)
    
    return waveform

def normalize_audio(audio):
    normalization_factor = torch.max(torch.abs(audio))
    if normalization_factor > 0:
        normalized_audio = audio / normalization_factor
    else:
        normalized_audio = audio
    return normalized_audio, normalization_factor


def process_audio(model, stft, istft, input_wav, device):
    input_wav_norm, norm_factor = normalize_audio(input_wav)
    
    with torch.no_grad():
        input_stft = stft(input_wav_norm)
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            enhanced_stft = model(input_stft)
        
        enhanced_wav = istft(enhanced_stft)
    
    if norm_factor > 0:
        enhanced_wav = enhanced_wav * norm_factor
    
    return enhanced_wav


def main():
    parser = argparse.ArgumentParser(
        description="Smule Renaissance Vocal Restoration"
    )
    parser.add_argument(
        "input", 
        type=str, 
        help="Input audio file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output audio file path (default: input_enhanced.wav)"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_enhanced.wav")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using device: CUDA with FP16 precision")
        dtype = torch.float16
    else:
        print("Using device: CPU with FP32 precision")
        dtype = torch.float32
    
    print(f"Loading model from {args.checkpoint}...")
    model = Renaissance().to(device).to(dtype)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    stft = STFT(n_fft=4096, hop_length=2048, win_length=4096)
    istft = iSTFT(n_fft=4096, hop_length=2048, win_length=4096)
    
    print(f"Loading audio from {args.input}...")
    input_wav = load_and_preprocess_audio(args.input, device, dtype)
    print(f"Audio duration: {input_wav.shape[1] / 48000:.2f} seconds")
    
    print("Processing audio...")
    enhanced_wav = process_audio(model, stft, istft, input_wav, device)
    
    print(f"Saving enhanced audio to {args.output}...")
    enhanced_wav_cpu = enhanced_wav.cpu().to(torch.float32)
    torchaudio.save(args.output, enhanced_wav_cpu, 48000)
    
    print("Done!")


if __name__ == "__main__":
    main()