#!/usr/bin/env python3

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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
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

class GradioInputDevice(agents.InputDevice):
    """Custom input device that works with Gradio"""
    
    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.is_active = True
    
    async def receive(self) -> agents.AudioChunk:
        """Receive audio data from the queue"""
        while self.is_active:
            try:
                audio_data = self.audio_queue.get(block=True, timeout=0.1)
                return audio_data
            except queue.Empty:
                await asyncio.sleep(0.1)
        
        return None
    
    def add_audio(self, audio_data):
        """Add audio data to the queue"""
        # Convert gradio audio format to AudioChunk
        sample_rate, audio_array = audio_data
        audio_chunk = agents.AudioChunk(
            samples=audio_array,
            sample_rate=sample_rate,
            is_last=False
        )
        self.audio_queue.put(audio_chunk)
    
    def stop(self):
        """Stop the input device"""
        self.is_active = False


class GradioOutputDevice(agents.OutputDevice):
    """Custom output device that works with Gradio"""
    
    def __init__(self):
        super().__init__()
        self.output_queue = queue.Queue()
    
    async def transmit(self, audio_chunk: agents.AudioChunk) -> None:
        """Transmit audio chunk to the queue"""
        if audio_chunk is not None:
            self.output_queue.put((audio_chunk.samples, audio_chunk.sample_rate))
    
    def get_latest_audio(self):
        """Get the latest audio from the queue"""
        try:
            return self.output_queue.get(block=False)
        except queue.Empty:
            return None


class SpeechPathologistAssistant:
    """Speech pathologist assistant using LiveKit agents and Gradio"""
    
    def __init__(self):
        self.input_device = GradioInputDevice()
        self.output_device = GradioOutputDevice()
        self.assistant = None
        self.assistant_task = None
        self.transcript = []
        self.is_running = False
    
    async def initialize_assistant(self):
        """Initialize the speech assistant"""
        self.assistant = agents.VoiceAssistant(
            openai_client=openai_client,
            model="gpt-4o",
            voice="shimmer",  # Using a friendly, professional voice
            input_device=self.input_device,
            output_device=self.output_device,
            initial_message=SPEECH_PATHOLOGIST_PROMPT,
            real_time=True,  # Enable real-time processing
        )
        
        # Add transcript callback
        self.assistant.on_transcript = self.on_transcript
        self.assistant.on_response = self.on_response
    
    def on_transcript(self, transcript):
        """Handle transcript from user"""
        self.transcript.append(f"Student: {transcript.text}")
        return True
    
    def on_response(self, response):
        """Handle response from assistant"""
        self.transcript.append(f"Speech Pathologist: {response.text}")
        return True
    
    async def start_assistant(self):
        """Start the assistant in a background task"""
        if not self.assistant:
            await self.initialize_assistant()
        
        self.is_running = True
        self.assistant_task = asyncio.create_task(self.assistant.run())
    
    def stop_assistant(self):
        """Stop the assistant"""
        if self.assistant_task and not self.assistant_task.done():
            self.assistant_task.cancel()
        
        self.input_device.stop()
        self.is_running = False
    
    def process_audio(self, audio):
        """Process audio from Gradio interface"""
        if not self.is_running or audio is None:
            return None, self.get_transcript()
        
        self.input_device.add_audio(audio)
        
        # Check for assistant output
        output_audio = self.output_device.get_latest_audio()
        
        return output_audio, self.get_transcript()
    
    def get_transcript(self):
        """Get the current transcript"""
        return "\n".join(self.transcript)


# Create the assistant instance
speech_assistant = SpeechPathologistAssistant()


def start_session():
    """Start the speech pathology session"""
    asyncio.run(speech_assistant.start_assistant())
    return "Session started. Please speak to begin the assessment."


def stop_session():
    """Stop the speech pathology session"""
    speech_assistant.stop_assistant()
    return "Session stopped."


def process_audio(audio):
    """Process audio from microphone"""
    if audio is None:
        return None, speech_assistant.get_transcript()
    
    return speech_assistant.process_audio(audio)


# Create Gradio Interface
with gr.Blocks(title="CASL-2 Speech Pathology Assistant") as app:
    gr.Markdown("# Speech Pathology Assistant")
    gr.Markdown("This tool provides speech therapy assessment based on the CASL-2 framework")
    
    with gr.Row():
        with gr.Column(scale=1):
            start_button = gr.Button("Start Session")
            stop_button = gr.Button("Stop Session")
            status = gr.Textbox(label="Status", value="Ready to start")
        
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                label="Speak",
                type="microphone", 
                streaming=True,
                autoplay=True
            )
            audio_output = gr.Audio(label="Response")
    
    with gr.Row():
        transcript = gr.Textbox(label="Transcript", lines=10)
    
    # Setup event handlers
    start_button.click(fn=start_session, outputs=status)
    stop_button.click(fn=stop_session, outputs=status)
    
    # Setup continuous audio processing
    audio_input.stream(
        fn=process_audio, 
        inputs=audio_input, 
        outputs=[audio_output, transcript]
    )


def main():
    """Main function to launch the Gradio app"""
    app.launch(share=True)


if __name__ == "__main__":
    main()