# Whisper AI: Advanced Audio Transcription, Translation, and Analysis

This Google Colaboratory notebook provides a comprehensive environment for leveraging OpenAI's Whisper model for various audio processing tasks. It goes beyond basic transcription to include features like language detection, translation, speaker diarization, and text summarization.

## Features:

* **Easy Setup**: Install all necessary libraries with a single cell.
* **Flexible Audio Input**: Upload files directly to Colab or use files from Google Drive.
* **Whisper Model Selection**: Choose from various Whisper models (tiny, base, small, medium, large, large-v2) to balance accuracy and performance.
* **Transcription**: Convert audio to text with automatic language detection.
* **Translation**: Translate transcribed text to English.
* **Speaker Diarization**: Identify and label different speakers in the audio (requires Hugging Face token).
* **Text Summarization**: Generate concise summaries of the transcribed content.
* **Structured Output**: Save results in various formats (TXT, SRT, VTT, JSON).

## Getting Started

To use this notebook, open it in Google Colab and run the cells sequentially.

### 1. Setup and Installation

This section installs all the necessary libraries, including `whisper`, `ffmpeg-python` (for audio processing), `pyannote.audio` (for speaker diarization), `transformers` (for summarization), and `pydub` (for audio manipulation).

```python
# @title Install Dependencies
!pip install -q git+[https://github.com/openai/whisper.git](https://github.com/openai/whisper.git)
!pip install -q ffmpeg-python
!pip install -q transformers
!pip install -q accelerate
!pip install -q sentencepiece # Required for some tokenizer models
!pip install -q pydub # For audio manipulation

# Install pyannote.audio for speaker diarization (requires specific versions)
# Note: pyannote.audio requires a Hugging Face token for models.
!pip install -q pyannote.audio==3.1.1
!pip install -q torchaudio==2.0.2

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torch Audio version: {torchaudio.__version__}")

import whisper
import os
from pydub import AudioSegment
from transformers import pipeline
import json
import re
from IPython.display import Audio, display
import numpy as np
import subprocess

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("All dependencies installed and imported successfully!")
