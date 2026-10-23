---
title: CASL 2 - Speech Therapy Assessment Tool
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# CASL Voice Bot

A speech pathology assistant using AI to assess students' speaking abilities based on the CASL-2 framework. This application helps speech-language pathologists (SLPs) with speech assessment in school settings.

## Implementations

This project provides multiple implementations:

1. **LiveKit Implementation** - Uses LiveKit agents with OpenAI's real-time voice API for low-latency, high-quality audio streaming.
2. **Direct API Implementation** - Uses OpenAI's API directly without LiveKit, for simpler deployment.
3. **Hugging Face Spaces** - An adaptive implementation that works on Hugging Face Spaces, automatically detecting whether LiveKit is available.

## Features

- Voice-to-voice interaction with AI speech pathologist
- CASL-2 framework assessment
- Real-time assessment tracking
- Session recording and saving
- Custom note-taking for SLPs
- Gradio web interface for easy sharing and use in school settings

## CASL-2 Assessment Areas

The AI speech pathologist assesses students in these key areas:

1. **Lexical/Semantic Skills**: Vocabulary knowledge, word meanings, and contextual word use
2. **Syntactic Skills**: Grammar and sentence structure understanding
3. **Supralinguistic Skills**: Higher-level language skills beyond literal meanings
4. **Pragmatic Skills**: Language use in social contexts (less emphasis for younger students)

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key with access to GPT-4o and TTS models

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/CASLVoiceBot.git
   cd CASLVoiceBot
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. For LiveKit implementation, install LiveKit dependencies:
   ```
   # Edit requirements.txt to uncomment the livekit-agents line
   pip install livekit-agents>=0.7.0
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit `.env` to add your OpenAI API key.

### Running the Application

#### LiveKit Implementation (recommended for best performance)
```
python run_livekit.py
```

#### Direct API Implementation (simpler deployment)
```
python run_direct.py
```

#### Command Line Options
Both implementations support these options:
- `--share`: Share the app publicly (enabled by default)
- `--local`: Run the app locally without sharing

## Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face with the Gradio SDK
2. Upload the repository contents to the Space
3. Add your OPENAI_API_KEY as a secret in the Space settings

By default, the Hugging Face Spaces deployment will try to use LiveKit if available, and fall back to direct API if not.

## Project Structure

```
CASLVoiceBot/
├── app.py                  # Hugging Face Spaces entry point
├── run_direct.py           # Direct API implementation runner
├── run_livekit.py          # LiveKit implementation runner
├── requirements.txt        # Common dependencies
├── .env.example            # Environment variables template
├── implementations/
│   ├── common/             # Shared utilities
│   │   ├── casl_utils.py   # CASL-2 assessment utilities
│   ├── direct/             # Direct API implementation
│   │   ├── app.py          # Direct OpenAI API app
│   ├── livekit/            # LiveKit implementation
│   │   ├── app.py          # LiveKit app
│   │   ├── livekit_gradio_hf.py # HF-compatible LiveKit app
├── session_data/           # Saved session data
```

## Usage

1. Optionally enter a Student ID to track sessions
2. Select your preferred AI voice
3. Click "Start Session" to begin a speech assessment
4. Wait for the AI to introduce itself, then speak when prompted
5. View real-time assessment in the interface
6. SLPs can add notes throughout the session
7. Save the session when finished
8. Click "Stop Session" to end

## License

[MIT License](LICENSE)