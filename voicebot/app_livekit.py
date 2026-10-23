#!/usr/bin/env python3

"""
CASL Voice Bot - Speech Pathology Assistant
Using LiveKit agents with OpenAI's real-time capabilities
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
        self.audio_queue = asyncio.Queue()
        self.is_active = True
    
    async def receive(self) -> agents.AudioChunk:
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
        sample_rate, audio_array = audio_data
        audio_chunk = agents.AudioChunk(
            samples=audio_array,
            sample_rate=sample_rate,
            is_last=False
        )
        await self.audio_queue.put(audio_chunk)
    
    def stop(self):
        """Stop the input device"""
        self.is_active = False


class GradioOutputDevice(agents.OutputDevice):
    """Custom output device that works with Gradio"""
    
    def __init__(self):
        super().__init__()
        self.output_queue = asyncio.Queue()
    
    async def transmit(self, audio_chunk: agents.AudioChunk) -> None:
        """Transmit audio chunk to the queue"""
        if audio_chunk is not None:
            await self.output_queue.put((audio_chunk.samples, audio_chunk.sample_rate))
    
    async def get_latest_audio(self):
        """Get the latest audio from the queue"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None


class SpeechPathologistAssistant:
    """Speech pathologist assistant using LiveKit agents"""
    
    def __init__(self):
        self.input_device = GradioInputDevice()
        self.output_device = GradioOutputDevice()
        self.assistant = None
        self.assistant_task = None
        self.transcript = []
        self.is_running = False
        self.notes = []
        self.current_assessment = {
            "lexical_semantic": 0,
            "syntactic": 0,
            "supralinguistic": 0,
            "pragmatic": 0
        }
    
    async def initialize_assistant(self, voice="shimmer"):
        """Initialize the speech assistant"""
        self.assistant = agents.VoiceAssistant(
            openai_client=openai_client,
            model="gpt-4o",
            voice=voice,
            input_device=self.input_device,
            output_device=self.output_device,
            initial_message=SPEECH_PATHOLOGIST_PROMPT,
            real_time=True,  # Enable real-time processing
        )
        
        # Add transcript and response callbacks
        self.assistant.on_transcript = self.on_transcript
        self.assistant.on_response = self.on_response
    
    def on_transcript(self, transcript):
        """Handle transcript from user"""
        self.transcript.append(f"Student: {transcript.text}")
        
        # Basic analysis of speech for CASL-2 categories
        self.analyze_speech(transcript.text)
        
        return True
    
    def on_response(self, response):
        """Handle response from assistant"""
        self.transcript.append(f"Speech Pathologist: {response.text}")
        return True
    
    def analyze_speech(self, text):
        """Analyze speech for CASL-2 categories"""
        # Simple heuristic analysis - in a real app, this would use more sophisticated NLP
        
        # Lexical/Semantic: check vocabulary diversity
        words = text.lower().split()
        unique_words = set(words)
        if len(unique_words) / max(1, len(words)) > 0.7:
            self.current_assessment["lexical_semantic"] += 1
            self.notes.append("Lexical/Semantic: Good vocabulary diversity")
        
        # Syntactic: check for sentence complexity
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        avg_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        if avg_words > 7:
            self.current_assessment["syntactic"] += 1
            self.notes.append("Syntactic: Complex sentence structures used")
        
        # Supralinguistic: check for figurative language (very basic check)
        figurative_markers = ["like", "as", "than", "seems", "appears", "metaphor", "imagine"]
        if any(marker in text.lower() for marker in figurative_markers):
            self.current_assessment["supralinguistic"] += 1
            self.notes.append("Supralinguistic: Potential figurative language detected")
        
        # Pragmatic: basic check for conversational elements
        pragmatic_markers = ["hello", "hi", "thanks", "thank you", "please", "excuse me", "sorry"]
        if any(marker in text.lower() for marker in pragmatic_markers):
            self.current_assessment["pragmatic"] += 1
            self.notes.append("Pragmatic: Appropriate social language detected")
    
    async def start_assistant(self, voice_model, student_id):
        """Start the assistant in a background task"""
        await self.initialize_assistant(voice_model)
        
        self.is_running = True
        
        # Add student info to transcript
        student_info = f" for {student_id}" if student_id else ""
        self.transcript.append(f"Session started{student_info}. The AI Speech Pathologist will speak first.")
        
        # Run the assistant in a background task
        self.assistant_task = asyncio.create_task(self.assistant.run())
        
        return "Session active. The AI will introduce itself."
    
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
            return None, self.get_transcript(), self.get_assessment_html()
        
        # Add audio to input device
        await self.input_device.add_audio(audio)
        
        # Check for assistant output
        output_audio = await self.output_device.get_latest_audio()
        
        return output_audio, self.get_transcript(), self.get_assessment_html()
    
    def get_transcript(self):
        """Get the current transcript"""
        return "\n".join(self.transcript)
    
    def get_assessment_html(self):
        """Get HTML representation of the current assessment"""
        html = """
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>CASL-2 Assessment Progress</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        """
        
        for category, value in self.current_assessment.items():
            category_name = category.replace('_', ' ').title()
            progress_width = min(100, value * 10)
            html += f"""
                <div style="padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div><strong>{category_name}</strong></div>
                    <div style="margin-top: 5px; height: 15px; background-color: #eee; border-radius: 7px; overflow: hidden;">
                        <div style="width: {progress_width}%; height: 100%; background-color: #4CAF50;"></div>
                    </div>
                    <div style="margin-top: 5px;">{value} points</div>
                </div>
            """
        
        html += """
            </div>
        """
        
        # Add recent notes
        if self.notes:
            html += """
            <div style="margin-top: 15px;">
                <h4>Recent Observations</h4>
                <ul style="margin-top: 5px;">
            """
            
            for note in self.notes[-5:]:
                html += f"<li>{note}</li>"
            
            html += """
                </ul>
            </div>
            """
        
        html += "</div>"
        return html
    
    def add_note(self, note):
        """Add a custom note"""
        if note.strip():
            self.notes.append(note)
            return f"Note added: {note}"
        return "Note was empty, not added."
    
    def save_session(self, student_id):
        """Save session to file"""
        if not student_id:
            student_id = "anonymous"
        
        # Create session data directory if it doesn't exist
        os.makedirs("session_data", exist_ok=True)
        
        # Save transcript
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"session_data/{student_id}_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write("\n".join(self.transcript))
            f.write("\n\n--- ASSESSMENT NOTES ---\n")
            for note in self.notes:
                f.write(f"- {note}\n")
            f.write("\n--- CASL-2 SCORES ---\n")
            for category, score in self.current_assessment.items():
                f.write(f"{category.replace('_', ' ').title()}: {score}\n")
        
        return f"Session saved to {filename}"


# Create the speech pathology assistant
speech_assistant = SpeechPathologistAssistant()


async def start_session(voice_model, student_id):
    """Start the speech pathology session"""
    status = await speech_assistant.start_assistant(voice_model, student_id)
    return status, None, speech_assistant.get_transcript(), speech_assistant.get_assessment_html()


def stop_session():
    """Stop the speech pathology session"""
    return speech_assistant.stop_assistant(), None, speech_assistant.get_transcript(), speech_assistant.get_assessment_html()


async def process_mic_input(audio, progress=gr.Progress()):
    """Process microphone input"""
    progress(0, desc="Processing speech...")
    audio_output, transcript, assessment = await speech_assistant.process_audio(audio)
    progress(1, desc="Done")
    return audio_output, transcript, assessment


def add_note(note):
    """Add a note to the session"""
    result = speech_assistant.add_note(note)
    return "", result, speech_assistant.get_assessment_html()


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
        7. SLPs can add assessment notes as needed
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