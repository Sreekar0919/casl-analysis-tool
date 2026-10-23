# CASL Voice Bot with LiveKit

A speech pathology assistant using LiveKit agents with OpenAI's real-time voice capabilities. This application helps speech-language pathologists (SLPs) assess students' speaking abilities based on the CASL-2 framework.

## Features

- Real-time voice interaction with AI speech pathologist using LiveKit
- OpenAI's GPT-4o for intelligent conversation
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
- Created using the LiveKit multimodal agent template

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
   pip install -r livekit_requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit `.env` to add your OpenAI API key.

### Running the Application

1. Start the application:
   ```
   python run_livekit.py
   ```

2. Access the application through the URL provided in the terminal.

## Usage

1. Optionally enter a Student ID to track sessions
2. Select your preferred AI voice
3. Click "Start Session" to begin a speech assessment
4. Wait for the AI to introduce itself, then speak when prompted
5. View real-time assessment in the interface
6. SLPs can add notes throughout the session
7. Save the session when finished
8. Click "Stop Session" to end

## Benefits of Using LiveKit

- **Real-time Audio Processing**: LiveKit provides robust real-time audio streaming capabilities
- **Low Latency**: Minimizes delay between student speech and AI response
- **WebRTC Infrastructure**: Built on the same technology used for video calls
- **Connection Management**: Automatically handles connection issues and reconnections
- **Scalability**: Can support multiple concurrent sessions if needed
- **Agent Integration**: LiveKit's agent system is designed specifically for AI assistants

## Deployment on Hugging Face Spaces

For deployment on Hugging Face Spaces, additional configuration may be required due to LiveKit's WebRTC requirements. Please refer to LiveKit documentation for details on setting up appropriate server configurations.

## License

[MIT License](LICENSE)