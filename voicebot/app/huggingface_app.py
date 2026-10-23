#!/usr/bin/env python3

"""
Hugging Face Spaces deployment file for CASL Voice Bot
This file is specifically designed for deploying on Hugging Face Spaces
"""

import os
import asyncio
import gradio as gr
import logging
import tempfile
import queue
import threading
import time
from dotenv import load_dotenv
from livekit import agents
from openai import AsyncOpenAI

# Load environment variables (will be set in HF Spaces secrets)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key from environment
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Speech Pathologist Agent Prompt
SPEECH_PATHOLOGIST_PROMPT = """
You are a speech pathologist, a healthcare professional who specializes in evaluating, diagnosing, and treating communication disorders, including speech, language, cognitive-communication, voice, and fluency disorders. Your role is to help patients improve their speech and communication skills through various therapeutic techniques and exercises.
Your are working with a student with speech impediments typically with ASD
You have to be rigid to help them stay on the right track. YOu have to start with some sort of intro activity and can not rely on teh student at all to complete your thoughts. You pick a place to start and assess teh speech from there. 
Each domain from the CASL-2 framework can be analyzed using the sample:
Lexical/Semantic Skills:
This category focuses on vocabulary knowledge, word meanings, and the ability to use words contextually. It measures both receptive and expressive language abilities related to word use.
Key Subtests:
Antonyms: Identifying words with opposite meanings.
Synonyms: Identifying words with similar meanings.
Idiomatic Language: Understanding and interpreting idioms and figurative language.
Evaluate vocabulary diversity (type-token ratio).
Note word-finding difficulties, incorrect word choices, or over-reliance on fillers (e.g., "like," "stuff").
Assess use of specific vs. vague language (e.g., "car" vs. "sedan").
Syntactic Skills:
This category evaluates understanding and use of grammar and sentence structure. It focuses on the ability to comprehend and produce grammatically correct sentences.
Key Subtests:
Sentence Expression: Producing grammatically correct sentences based on prompts.
Grammaticality Judgment: Identifying whether a sentence is grammatically correct.
Examine sentence structure for grammatical accuracy.
Identify errors in verb tense, subject-verb agreement, or sentence complexity.
Note the use of clauses, conjunctions, and varied sentence types.
Supralinguistic Skills:
This subcategory assesses higher-level language skills that go beyond literal meanings, such as understanding implied meanings, sarcasm, and complex verbal reasoning.
Key Subtests:
Inferences: Understanding information that is not explicitly stated.
Meaning from Context: Deriving meaning from surrounding text or dialogue.
Nonliteral Language: Interpreting figurative language, such as metaphors or irony
Look for use or understanding of figurative language, idioms, or humor.
Assess ability to handle ambiguous or implied meanings in context.
Identify advanced language use for abstract or hypothetical ideas.
Pragmatic Skills(focus less on this as it is not typically necessary for the age range you will be dealing with):
This category measures the ability to use language effectively in social contexts. It evaluates understanding of conversational rules, turn-taking, and adapting communication to different social situations.
Key Subtests:
Pragmatic Language Test: Assessing appropriateness of responses in social scenarios.

Begin by introducing yourself as a speech pathologist and start with a simple vocabulary assessment activity. Be encouraging but structured in your approach.
"""

# Custom audio processing for Gradio interface
class AudioProcessor:
    def __init__(self):
        self.transcript = []
        self.is_active = False
        self.voice_model = "shimmer"  # Default voice
    
    async def process_speech(self, audio_data, openai_client):
        """Process speech using OpenAI's API"""
        if not self.is_active or audio_data is None:
            return None, "\n".join(self.transcript)
            
        # Prepare audio file for OpenAI
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        
        try:
            # Save audio data to temporary file
            sample_rate, audio_array = audio_data
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
                
                # Generate assistant response
                chat_response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SPEECH_PATHOLOGIST_PROMPT},
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
                
                return (sample_rate, audio_data), "\n".join(self.transcript)
                
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            self.transcript.append(f"Error: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_file.name)
        
        return None, "\n".join(self.transcript)
    
    def start_session(self, voice_model):
        """Start a new session"""
        self.is_active = True
        self.voice_model = voice_model if voice_model else "shimmer"
        self.transcript = []
        self.transcript.append("Session started. The AI Speech Pathologist will speak first.")
        return "Session active. Please wait for the AI to introduce itself."
    
    def stop_session(self):
        """Stop the current session"""
        self.is_active = False
        return "Session stopped."


# Create the audio processor instance
audio_processor = AudioProcessor()


async def start_session(voice_model):
    """Start the speech pathology session"""
    status = audio_processor.start_session(voice_model)
    
    # Generate initial AI introduction
    try:
        # Generate assistant response
        chat_response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SPEECH_PATHOLOGIST_PROMPT},
                {"role": "user", "content": "Hello"}  # Initial trigger
            ]
        )
        
        assistant_text = chat_response.choices[0].message.content
        audio_processor.transcript.append(f"Speech Pathologist: {assistant_text}")
        
        # Generate speech from text
        speech_response = await openai_client.audio.speech.create(
            model="tts-1",
            voice=audio_processor.voice_model,
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
        
        return status, (sample_rate, audio_data), "\n".join(audio_processor.transcript)
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        audio_processor.transcript.append(f"Error: {str(e)}")
        return status, None, "\n".join(audio_processor.transcript)


def stop_session():
    """Stop the speech pathology session"""
    return audio_processor.stop_session(), None, "\n".join(audio_processor.transcript)


async def process_mic_input(audio, progress=gr.Progress()):
    """Process microphone input"""
    if audio is None or not audio_processor.is_active:
        return None, "\n".join(audio_processor.transcript)
    
    progress(0, desc="Processing speech...")
    audio_output, transcript = await audio_processor.process_speech(audio, openai_client)
    progress(1, desc="Done")
    return audio_output, transcript


# Create Gradio Interface
with gr.Blocks(title="CASL-2 Speech Pathology Assistant") as app:
    gr.Markdown("# CASL-2 Speech Pathology Assistant")
    gr.Markdown("### AI-powered speech therapy assessment based on the CASL-2 framework")
    
    with gr.Row():
        with gr.Column(scale=1):
            voice_select = gr.Dropdown(
                ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], 
                value="shimmer", 
                label="Assistant Voice"
            )
            start_button = gr.Button("Start Session", variant="primary")
            stop_button = gr.Button("Stop Session", variant="stop")
            status = gr.Textbox(label="Status", value="Ready to start")
        
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="AI Speech", autoplay=True)
            audio_input = gr.Audio(
                label="Speak to the AI",
                type="microphone",
                source="microphone", 
                streaming=True
            )
    
    with gr.Row():
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
        
        1. Select the AI voice you prefer
        2. Click "Start Session" to begin
        3. The AI will introduce itself and begin the assessment
        4. Speak into your microphone when it's your turn
        5. View the transcript to track the conversation
        6. Click "Stop Session" when finished
        
        ### For Speech-Language Pathologists
        
        This tool is designed to supplement, not replace, professional SLP services. The source code is available for customization to meet specific assessment needs.
        """)
    
    # Setup event handlers
    start_button.click(
        fn=lambda voice: asyncio.run(start_session(voice)), 
        inputs=voice_select,
        outputs=[status, audio_output, transcript]
    )
    stop_button.click(fn=stop_session, outputs=[status, audio_output, transcript])
    
    # Setup audio processing
    audio_input.stream(
        fn=lambda audio: asyncio.run(process_mic_input(audio)),
        inputs=audio_input,
        outputs=[audio_output, transcript]
    )

# Launch the app
app.launch(share=True)