#!/usr/bin/env python3

"""
Hugging Face Spaces entry point for CASL Voice Bot
"""

# Import the app from app_main.py
from app_main import app

# This is the entry point that Hugging Face Spaces will use
if __name__ == "__main__":
    app.launch()