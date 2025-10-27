# Smule Renaissance Small

A 10.4M paramater generative audio model for restoring degraded vocals in any situation that runs 10.5x faster than real-time on iPhone 12's CPU;
Outperforms all open source models in subjective quality; matches commericial models on singing voice restoration.

Technical Report: [![Technical Report](https://img.shields.io/badge/arXiv-2510.21659-blue.svg)](https://arxiv.org/abs/2510.21659)

HuggingFace Model: [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-SmuleRenaissanceSmall-yellow.svg)](https://huggingface.co/smulelabs/Smule-Renaissance-Small)

Extreme Degradation Bench: [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-ExtremeDegradationBench-green.svg)](https://huggingface.co/datasets/smulelabs/ExtremeDegradationBench)

---

## Getting Started
### Setting up environment
```bash
# Create a virtual environment
uv venv cleanup --python=3.10
source cleanup/bin/activate
uv pip install -r requirements.txt
```

Download the model checkpoint from [HuggingFace](https://huggingface.co/smulelabs/Smule-Renaissance-Small) and place it in the root directory.
```bash
wget https://huggingface.co/smulelabs/Smule-Renaissance-Small/resolve/main/smule-renaissance-small.pt
```

### Running the model
```bash
python main.py {path-to-input} -o {path-to-output} -c {path-to-checkpoint}
```