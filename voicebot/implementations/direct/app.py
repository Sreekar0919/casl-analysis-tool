#!/usr/bin/env python3

"""
CASL Voice Bot - Speech Pathology Assistant
Direct OpenAI API implementation (no LiveKit)
"""

import os
import asyncio
import gradio as gr
import logging
import sys
import tempfile
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

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


class SpeechPathologistAssistant:
    """Speech pathologist assistant using direct OpenAI API"""
    
    def __init__(self):
        self.transcript = []
        self.is_running = False
        self.assessment = CASLAssessment()
        self.voice_model = "shimmer"
        self.student_id = None
    
    async def start_session(self, voice_model, student_id):
        """Start a new session"""
        self.is_running = True
        self.voice_model = voice_model if voice_model else "shimmer"
        self.student_id = student_id
        self.transcript = []
        self.assessment = CASLAssessment()
        
        # Add student info to transcript
        student_info = f" for {student_id}" if student_id else ""
        self.transcript.append(f"Session started{student_info}. The AI Speech Pathologist will speak first.")
        
        # Generate initial AI message
        initial_audio = await self.generate_initial_message()
        
        return "Session active. The AI will introduce itself.", initial_audio, self.get_transcript(), self.assessment.get_assessment_html()
    
    async def generate_initial_message(self):
        """Generate initial AI message"""
        # Generate assistant response
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
        
        # Save speech to temporary file
        response_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response_temp_file.close()
        
        speech_response.stream_to_file(response_temp_file.name)
        
        # Load audio data for Gradio
        import soundfile as sf
        audio_data, sample_rate = sf.read(response_temp_file.name)
        
        # Clean up
        os.unlink(response_temp_file.name)
        
        return (sample_rate, audio_data)
    
    def stop_session(self):
        """Stop the current session"""
        self.is_running = False
        self.transcript.append("Session ended.")
        return "Session stopped.", None, self.get_transcript(), self.assessment.get_assessment_html()
    
    async def process_audio(self, audio):
        """Process audio from Gradio interface"""
        if not self.is_running or audio is None:
            return None, self.get_transcript(), self.assessment.get_assessment_html()
        
        # Prepare audio file for OpenAI
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        
        try:
            # Save audio data to temporary file
            sample_rate, audio_array = audio
            import scipy.io.wavfile
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
                response_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                response_temp_file.close()
                
                speech_response.stream_to_file(response_temp_file.name)
                
                # Load audio data for Gradio
                import soundfile as sf
                audio_data, sample_rate = sf.read(response_temp_file.name)
                
                # Clean up
                os.unlink(response_temp_file.name)
                
                return (sample_rate, audio_data), self.get_transcript(), self.assessment.get_assessment_html()
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self.transcript.append(f"Error: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_file.name)
        
        return None, self.get_transcript(), self.assessment.get_assessment_html()
    
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
    return await speech_assistant.start_session(voice_model, student_id)


def stop_session():
    """Stop the speech pathology session"""
    return speech_assistant.stop_session()


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


def main(share=True):
    """Main function to launch the app"""
    app.launch(share=share)


# Entry point for the application
if __name__ == "__main__":
    main()