#!/usr/bin/env python3

"""
CASL Voice Bot - Hugging Face Spaces entry point
"""

# Import the adaptive app that works in both LiveKit and direct modes
from implementations.livekit.livekit_gradio_hf import app

# This is the entry point that Hugging Face Spaces will use
if __name__ == "__main__":
    app.launch()