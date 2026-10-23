#!/usr/bin/env python3

"""
Run the CASL Voice Bot using LiveKit and OpenAI real-time capabilities
"""

import argparse
from implementations.livekit.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASL Voice Bot with LiveKit - AI Speech Therapist")
    parser.add_argument("--share", action="store_true", help="Share the app publicly")
    parser.add_argument("--local", action="store_true", help="Run the app locally without sharing")
    args = parser.parse_args()
    
    main(share=not args.local)