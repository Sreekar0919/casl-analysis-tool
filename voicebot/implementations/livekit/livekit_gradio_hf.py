#!/usr/bin/env python3

"""
CASL Voice Bot - Hugging Face Spaces deployment version
Using LiveKit with Gradio for Hugging Face Spaces compatibility

This is a special version optimized for Hugging Face Spaces deployment
that works with LiveKit's WebRTC capabilities.
"""

import os
import asyncio
import gradio as gr
import logging
import sys
import time
import importlib.util
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Check if livekit-agents is installed
try:
    from livekit import agents
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    # Create dummy classes for type hinting
    class agents:
        class InputDevice: pass
        class OutputDevice: pass
        class AudioChunk: pass
        class VoiceAssistant: pass

# Add parent directory to path to import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementations.common.casl_utils import CASLAssessment, save_session_data, CASL_PROMPT

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Hugging Face Spaces compatibility check
HF_SPACES = os.environ.get("SPACE_ID") is not None
logger.info(f"Running in Hugging Face Spaces: {HF_SPACES}")
logger.info(f"LiveKit available: {LIVEKIT_AVAILABLE}")

class GradioInputDevice(agents.InputDevice if LIVEKIT_AVAILABLE else object):
    """Custom input device that works with Gradio"""
    
    def __init__(self):
        if LIVEKIT_AVAILABLE:
            super().__init__()
        self.audio_queue = asyncio.Queue()
        self.is_active = True
    
    async def receive(self):
        """Receive audio data from the queue"""
        try:
            audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            return audio_data
        except asyncio.TimeoutError:
            return None
    
    async def add_audio(self, audio_data):
        """Add audio data to the queue"""
        if audio_data is None:
            return
            
        # Convert gradio audio format to AudioChunk
        if LIVEKIT_AVAILABLE:
            sample_rate, audio_array = audio_data
            audio_chunk = agents.AudioChunk(
                samples=audio_array,
                sample_rate=sample_rate,
                is_last=False
            )
            await self.audio_queue.put(audio_chunk)
        else:
            # Store raw audio data if LiveKit is not available
            await self.audio_queue.put(audio_data)
    
    def stop(self):
        """Stop the input device"""
        self.is_active = False


class GradioOutputDevice(agents.OutputDevice if LIVEKIT_AVAILABLE else object):
    """Custom output device that works with Gradio"""
    
    def __init__(self):
        if LIVEKIT_AVAILABLE:
            super().__init__()
        self.output_queue = asyncio.Queue()
    
    async def transmit(self, audio_chunk):
        """Transmit audio chunk to the queue"""
        if audio_chunk is not None:
            if LIVEKIT_AVAILABLE:
                await self.output_queue.put((audio_chunk.samples, audio_chunk.sample_rate))
            else:
                # Handle raw audio data if LiveKit is not available
                await self.output_queue.put(audio_chunk)
    
    async def get_latest_audio(self):
        """Get the latest audio from the queue"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None


class SpeechPathologistAssistant:
    """Speech pathologist assistant using LiveKit agents or direct OpenAI API"""
    
    def __init__(self):
        self.input_device = GradioInputDevice()
        self.output_device = GradioOutputDevice()
        self.assistant = None
        self.assistant_task = None
        self.transcript = []
        self.is_running = False
        self.assessment = CASLAssessment()
        self.student_id = None
        self.voice_model = "shimmer"
    
    async def initialize_assistant(self, voice="shimmer"):
        """Initialize the speech assistant"""
        self.voice_model = voice
        
        if LIVEKIT_AVAILABLE:
            # Use LiveKit VoiceAssistant if available
            self.assistant = agents.VoiceAssistant(
                openai_client=openai_client,
                model="gpt-4o",
                voice=voice,
                input_device=self.input_device,
                output_device=self.output_device,
                initial_message=CASL_PROMPT,
                real_time=True,  # Enable real-time processing
            )
            
            # Add transcript and response callbacks
            self.assistant.on_transcript = self.on_transcript
            self.assistant.on_response = self.on_response
        else:
            # If LiveKit is not available, we'll use direct OpenAI API
            logger.info("LiveKit not available, using direct OpenAI API")
    
    def on_transcript(self, transcript):
        """Handle transcript from user (for LiveKit)"""
        self.transcript.append(f"Student: {transcript.text}")
        
        # Basic analysis of speech for CASL-2 categories
        self.assessment.analyze_speech(transcript.text)
        
        return True
    
    def on_response(self, response):
        """Handle response from assistant (for LiveKit)"""
        self.transcript.append(f"Speech Pathologist: {response.text}")
        return True
    
    async def start_assistant(self, voice_model, student_id):
        """Start the assistant in a background task"""
        self.student_id = student_id
        await self.initialize_assistant(voice_model)
        
        self.is_running = True
        
        # Add student info to transcript
        student_info = f" for {student_id}" if student_id else ""
        self.transcript.append(f"Session started{student_info}. The AI Speech Pathologist will speak first.")
        
        if LIVEKIT_AVAILABLE:
            # Run the LiveKit assistant in a background task
            self.assistant_task = asyncio.create_task(self.assistant.run())
        else:
            # For direct OpenAI API, generate initial message
            await self.generate_initial_message()
        
        return "Session active. The AI will introduce itself."
    
    async def generate_initial_message(self):
        """Generate initial AI message when not using LiveKit"""
        # Generate assistant response using OpenAI API directly
        chat_response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CASL_PROMPT},
                {"role": "user", "content": "Hello"}  # Initial trigger
            ]
        )
        
        assistant_text = chat_response.choices[0].message.content
        self.transcript.append(f"Speech Pathologist: {assistant_text}")
        
        # Generate speech from text
        speech_response = await openai_client.audio.speech.create(
            model="tts-1",
            voice=self.voice_model,
            input=assistant_text
        )
        
        # Save speech to temporary file to get audio data
        import tempfile
        import soundfile as sf
        
        response_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response_temp_file.close()
        
        speech_response.stream_to_file(response_temp_file.name)
        
        # Load audio data for Gradio
        audio_data, sample_rate = sf.read(response_temp_file.name)
        
        # Send to output device
        await self.output_device.transmit((audio_data, sample_rate))
        
        # Clean up
        os.unlink(response_temp_file.name)
    
    def stop_assistant(self):
        """Stop the assistant"""
        if self.assistant_task and not self.assistant_task.done():
            self.assistant_task.cancel()
        
        self.input_device.stop()
        self.is_running = False
        
        # Add ending to transcript
        self.transcript.append("Session ended.")
        return "Session stopped."
    
    async def process_audio(self, audio):
        """Process audio from Gradio interface"""
        if not self.is_running or audio is None:
            return None, self.get_transcript(), self.assessment.get_assessment_html()
        
        if LIVEKIT_AVAILABLE:
            # Add audio to input device for LiveKit
            await self.input_device.add_audio(audio)
            
            # Check for assistant output
            output_audio = await self.output_device.get_latest_audio()
        else:
            # For direct OpenAI API
            output_audio = await self.process_with_direct_api(audio)
        
        return output_audio, self.get_transcript(), self.assessment.get_assessment_html()
    
    async def process_with_direct_api(self, audio):
        """Process audio using direct OpenAI API when LiveKit is not available"""
        # Prepare audio file for OpenAI
        import tempfile
        import scipy.io.wavfile
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        
        try:
            # Save audio data to temporary file
            sample_rate, audio_array = audio
            scipy.io.wavfile.write(temp_file.name, sample_rate, audio_array)
            
            # Transcribe audio using OpenAI
            with open(temp_file.name, "rb") as audio_file:
                transcript_response = await openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
            
            user_text = transcript_response.text
            if user_text.strip():
                self.transcript.append(f"Student: {user_text}")
                
                # Analyze speech for CASL-2 categories
                self.assessment.analyze_speech(user_text)
                
                # Generate assistant response
                chat_response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": CASL_PROMPT},
                        {"role": "user", "content": user_text}
                    ]
                )
                
                assistant_text = chat_response.choices[0].message.content
                self.transcript.append(f"Speech Pathologist: {assistant_text}")
                
                # Generate speech from text
                speech_response = await openai_client.audio.speech.create(
                    model="tts-1",
                    voice=self.voice_model,
                    input=assistant_text
                )
                
                # Save speech to temporary file
                import soundfile as sf
                
                response_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                response_temp_file.close()
                
                speech_response.stream_to_file(response_temp_file.name)
                
                # Load audio data for Gradio
                audio_data, sample_rate = sf.read(response_temp_file.name)
                
                # Clean up
                os.unlink(response_temp_file.name)
                
                return (sample_rate, audio_data)
        except Exception as e:
            logger.error(f"Error processing with direct API: {e}")
        finally:
            # Clean up temp file
            os.unlink(temp_file.name)
        
        return None
    
    def get_transcript(self):
        """Get the current transcript"""
        return "\n".join(self.transcript)
    
    def add_note(self, note):
        """Add a custom note"""
        result = self.assessment.add_note(note)
        return "", result, self.assessment.get_assessment_html()
    
    def save_session(self, student_id=None):
        """Save session to file"""
        student_id = student_id or self.student_id
        return save_session_data(self.transcript, self.assessment, student_id)


# Create the speech pathology assistant
speech_assistant = SpeechPathologistAssistant()


async def start_session(voice_model, student_id):
    """Start the speech pathology session"""
    status = await speech_assistant.start_assistant(voice_model, student_id)
    return status, None, speech_assistant.get_transcript(), speech_assistant.assessment.get_assessment_html()


def stop_session():
    """Stop the speech pathology session"""
    return speech_assistant.stop_assistant(), None, speech_assistant.get_transcript(), speech_assistant.assessment.get_assessment_html()


async def process_mic_input(audio, progress=gr.Progress()):
    """Process microphone input"""
    progress(0, desc="Processing speech...")
    audio_output, transcript, assessment = await speech_assistant.process_audio(audio)
    progress(1, desc="Done")
    return audio_output, transcript, assessment


def add_note(note):
    """Add a note to the session"""
    return speech_assistant.add_note(note)


def save_session(student_id):
    """Save the current session"""
    return speech_assistant.save_session(student_id)


# Create Gradio Interface
with gr.Blocks(title="CASL-2 Speech Pathology Assistant") as app:
    gr.Markdown("# CASL-2 Speech Pathology Assistant")
    gr.Markdown("### AI-powered speech therapy assessment based on the CASL-2 framework")
    
    # Show LiveKit availability
    if not LIVEKIT_AVAILABLE:
        gr.Markdown(
            "⚠️ **Notice:** Running without LiveKit agents. Using direct OpenAI API instead. "
            "For best performance, install livekit-agents package."
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            student_id = gr.Textbox(label="Student ID (optional)", placeholder="Enter student ID")
            voice_select = gr.Dropdown(
                ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], 
                value="shimmer", 
                label="Assistant Voice"
            )
            start_button = gr.Button("Start Session", variant="primary")
            stop_button = gr.Button("Stop Session", variant="stop")
            status = gr.Textbox(label="Status", value="Ready to start")
            
            with gr.Accordion("SLP Tools", open=True):
                note_input = gr.Textbox(
                    label="Add Assessment Note", 
                    placeholder="Enter observation or assessment note here..."
                )
                note_button = gr.Button("Add Note")
                note_status = gr.Textbox(label="Note Status")
                save_button = gr.Button("Save Session")
                save_status = gr.Textbox(label="Save Status")
        
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="AI Speech", autoplay=True)
            audio_input = gr.Audio(
                label="Speak to the AI",
                type="microphone",
                source="microphone", 
                streaming=True
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            assessment_html = gr.HTML(label="Assessment Progress")
        with gr.Column(scale=1):
            transcript = gr.Textbox(label="Transcript", lines=10)
    
    with gr.Accordion("About This Application", open=False):
        gr.Markdown("""
        ### About CASL-2 Speech Pathology Assistant
        
        This application provides an AI speech pathologist that can assess students using the CASL-2 framework. It focuses on:
        
        - **Lexical/Semantic Skills**: Vocabulary knowledge and word usage
        - **Syntactic Skills**: Grammar and sentence structure
        - **Supralinguistic Skills**: Higher-level language beyond literal meanings
        - **Pragmatic Skills**: Social use of language (less emphasis for younger students)
        
        The AI will provide structured assessments and exercises to help evaluate speech patterns.
        
        ### How to Use
        
        1. Optionally enter a Student ID to track sessions
        2. Select the AI voice you prefer
        3. Click "Start Session" to begin
        4. The AI will introduce itself and begin the assessment
        5. Speak into your microphone when it's your turn
        6. View the transcript to track the conversation
        7. SLPs can add notes throughout the session
        8. Save the session when finished
        9. Click "Stop Session" when done
        
        ### For Speech-Language Pathologists
        
        This tool is designed to supplement, not replace, professional SLP services. SLPs can:
        
        - Add custom notes during the session
        - Save session data for later reference
        - Track progress across multiple sessions
        - Use the AI as a consistent assessment tool
        """)
    
    # Setup event handlers
    start_button.click(
        fn=lambda voice, student: asyncio.run(start_session(voice, student)), 
        inputs=[voice_select, student_id],
        outputs=[status, audio_output, transcript, assessment_html]
    )
    stop_button.click(
        fn=stop_session, 
        outputs=[status, audio_output, transcript, assessment_html]
    )
    note_button.click(
        fn=add_note,
        inputs=note_input,
        outputs=[note_input, note_status, assessment_html]
    )
    save_button.click(
        fn=save_session,
        inputs=student_id,
        outputs=save_status
    )
    
    # Setup audio processing
    audio_input.stream(
        fn=lambda audio: asyncio.run(process_mic_input(audio)),
        inputs=audio_input,
        outputs=[audio_output, transcript, assessment_html]
    )


# Entry point for the application
if __name__ == "__main__":
    app.launch(share=True)