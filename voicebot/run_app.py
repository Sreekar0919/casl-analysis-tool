#!/usr/bin/env python3

"""
Main entry point for the CASL Voice Bot application.
Launches the Gradio web interface.
"""

import argparse
from app_main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASL Voice Bot - AI Speech Therapist")
    parser.add_argument("--share", action="store_true", help="Share the app publicly")
    parser.add_argument("--local", action="store_true", help="Run the app locally without sharing")
    args = parser.parse_args()
    
    main(share=not args.local)