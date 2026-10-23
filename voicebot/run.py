#!/usr/bin/env python3

"""
Main entry point for the CASL Voice Bot application.
Launches the Gradio web interface.
"""

import argparse
from app.gradio_app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASL Voice Bot - AI Speech Therapist")
    parser.add_argument("--share", action="store_true", help="Share the app publicly (for HuggingFace)")
    args = parser.parse_args()
    
    main()