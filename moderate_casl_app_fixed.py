import gradio as gr
import json
import os
import logging
import requests
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anthropic API key - can be set as HuggingFace secret or environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Check if API key is available
if ANTHROPIC_API_KEY:
    logger.info("Claude API key found")
else:
    logger.warning("Claude API key not found - using demo mode")

def call_claude_api(prompt):
    """Call Claude API directly"""
    if not ANTHROPIC_API_KEY:
        return "❌ Claude API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            response_json = response.json()
            return response_json['content'][0]['text']
        else:
            logger.error(f"Claude API error: {response.status_code} - {response.text}")
            return f"❌ Claude API Error: {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return f"❌ Error: {str(e)}"

def process_file(file):
    """Process uploaded file"""
    if file is None:
        return "Please upload a file first."
    
    try:
        # Read file content
        with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return "File appears to be empty."
            
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def read_cha_file(file_path):
    """Read and parse a .cha transcript file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract participant lines (starting with *PAR:)
        par_lines = []
        for line in content.splitlines():
            if line.startswith('*PAR:'):
                par_lines.append(line)
                
        # If no PAR lines found, just return the whole content
        if not par_lines:
            return content
            
        return '\n'.join(par_lines)
    
    except Exception as e:
        logger.error(f"Error reading CHA file: {str(e)}")
        return ""

def process_upload(file):
    """Process an uploaded file (text or CHA)"""
    if file is None:
        return ""
    
    file_path = file.name
    if file_path.endswith('.cha'):
        return read_cha_file(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def generate_demo_response(prompt):
    """Generate a demo response when API is not available"""
    return """## Speech Factors Analysis

**Difficulty producing fluent speech**: 8 instances, moderate severity
- Examples: "today I would &-um like to talk about &-um a fun trip"
- "we went to the &-um &-um beach [//] no to the mountains [//] I mean the beach actually"

**Word retrieval issues**: 6 instances, mild-moderate severity  
- Examples: "what do you call those &-um &-um sprinkles! that's the word"
- "sometimes I forget [//] forgetted [: forgot] [*] what they call those things we built"

**Grammatical errors**: 4 instances, moderate severity
- Examples: "after swimming we [//] I eat [: ate] [*] &-um ice cream"
- "we saw [/] saw fishies [: fish] [*] swimming in the water"

**Repetitions and revisions**: 5 instances, mild severity
- Examples: "we [/] we stayed for &-um three no [//] four days"
- "I want to go back to the beach [/] beach next year"

## Language Skills Assessment

**Lexical/Semantic Skills**: 
- Vocabulary diversity appears age-appropriate with some word-finding difficulties
- Examples: "what do you call those &-um &-um sprinkles! that's the word"
- Shows good semantic understanding but retrieval challenges

**Syntactic Skills**:
- Basic sentence structure is intact with some grammatical inconsistencies
- Examples: "my brother he [//] he helped me dig a big hole"
- Verb tense errors noted: "forgetted" for "forgot", "eat" for "ate"

**Supralinguistic Skills**:
- Narrative organization is good with logical sequence
- Examples: "sometimes I wonder [/] wonder where fishies [: fish] [*] go when it's cold"
- Shows creative thinking and topic maintenance

## Treatment Recommendations

1. **Word-finding strategies**: Implement semantic cuing techniques using the patient's experiences (beach, ice cream) as context
2. **Grammar practice**: Focus on verb tense consistency with structured exercises
3. **Fluency techniques**: Work on reducing fillers and improving speech flow
4. **Self-monitoring**: Help patient identify and correct grammatical errors
5. **Vocabulary expansion**: Build on existing semantic networks

## Clinical Summary

This child demonstrates a mild-to-moderate expressive language disorder with primary concerns in word retrieval and grammatical accuracy. Strengths include good narrative organization and topic maintenance. The pattern suggests intervention should focus on word-finding strategies and grammatical form practice while building on existing semantic knowledge."""

def analyze_transcript(transcript, age, gender, slp_notes=""):
    """Analyze a speech transcript using Claude"""
    if not transcript or len(transcript.strip()) < 50:
        return "Error: Please provide a longer transcript for analysis."
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    # Simplified analysis prompt
    prompt = f"""
    You are a speech-language pathologist analyzing a transcript for CASL assessment.
    
    Patient: {age}-year-old {gender}
    
    TRANSCRIPT:
    {transcript}{notes_section}
    
    Please provide a comprehensive CASL analysis including:
    
    1. SPEECH FACTORS (with counts and severity):
    - Difficulty producing fluent speech
    - Word retrieval issues  
    - Grammatical errors
    - Repetitions and revisions
    
    2. LANGUAGE SKILLS ASSESSMENT:
    - Lexical/Semantic Skills (qualitative assessment)
    - Syntactic Skills (qualitative assessment)
    - Supralinguistic Skills (qualitative assessment)
    
    3. TREATMENT RECOMMENDATIONS:
    - List 3-5 specific intervention strategies
    
    4. CLINICAL SUMMARY:
    - Brief explanation of findings and prognosis
    
    Use exact quotes from the transcript as evidence.
    Focus on qualitative observations rather than standardized scores.
    Be specific and provide concrete examples from the transcript.
    {f"Consider the SLP clinical notes in your analysis." if slp_notes and slp_notes.strip() else ""}
    """
    
    # Get analysis from Claude API or demo
    if ANTHROPIC_API_KEY:
        result = call_claude_api(prompt)
    else:
        result = generate_demo_response(prompt)
    
    return result

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="Enhanced CASL Analysis Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🗣️ Enhanced CASL Analysis Tool")
        gr.Markdown("Upload a speech transcript and get comprehensive CASL assessment results.")
        
        with gr.Tabs():
            # Analysis Tab
            with gr.Tab("📊 Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Patient Information")
                        
                        with gr.Row():
                            age = gr.Number(label="Age", value=8, minimum=1, maximum=120)
                            gender = gr.Radio(["male", "female", "other"], label="Gender", value="male")
                        
                        slp_notes = gr.Textbox(
                            label="SLP Clinical Notes (Optional)",
                            placeholder="Enter any additional clinical observations, context, or notes...",
                            lines=3
                        )
                        
                        gr.Markdown("### Transcript Input")
                        
                        file_upload = gr.File(
                            label="Upload Transcript File",
                            file_types=[".txt", ".cha"]
                        )
                        
                        transcript = gr.Textbox(
                            label="Or Paste Transcript Here",
                            placeholder="Enter transcript text or upload a file...",
                            lines=10
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Transcript", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Analysis Results")
                        
                        analysis_output = gr.Textbox(
                            label="CASL Analysis Report",
                            placeholder="Analysis results will appear here...",
                            lines=25,
                            max_lines=30
                        )
            
            # Sample Transcripts Tab
            with gr.Tab("📝 Sample Transcripts"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Sample Transcripts")
                        
                        sample_choice = gr.Dropdown(
                            choices=[
                                "Beach Trip (Child)",
                                "School Day (Adolescent)", 
                                "Adult Recovery"
                            ],
                            label="Select a sample transcript:",
                            value="Beach Trip (Child)"
                        )
                        
                        load_sample_btn = gr.Button("Load Sample", variant="secondary")
                        
                        sample_transcript = gr.Textbox(
                            label="Sample Transcript",
                            lines=15,
                            interactive=False
                        )
                        
                        use_sample_btn = gr.Button("Use This Sample for Analysis", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Sample Descriptions")
                        
                        gr.Markdown("""
                        **Beach Trip (Child)**: 8-year-old child describing a family beach vacation
                        - Shows typical child language patterns
                        - Contains word-finding difficulties and grammatical errors
                        - Good narrative structure despite language challenges
                        
                        **School Day (Adolescent)**: Teenager describing a school day
                        - More complex language but still some disfluencies
                        - Shows adolescent speech patterns
                        - Academic vocabulary and social language
                        
                        **Adult Recovery**: Adult describing stroke recovery
                        - Post-stroke language patterns
                        - Word-finding difficulties
                        - Shows recovery progress
                        """)
        
        # Sample transcripts
        SAMPLE_TRANSCRIPTS = {
            "Beach Trip (Child)": """*PAR: today I would &-um like to talk about &-um a fun trip I took last &-um summer with my family.
*PAR: we went to the &-um &-um beach [//] no to the mountains [//] I mean the beach actually.
*PAR: there was lots of &-um &-um swimming and &-um sun.
*PAR: we [/] we stayed for &-um three no [//] four days in a &-um hotel near the water [: ocean] [*].
*PAR: my favorite part was &-um building &-um castles with sand.
*PAR: sometimes I forget [//] forgetted [: forgot] [*] what they call those things we built.
*PAR: my brother he [//] he helped me dig a big hole.
*PAR: we saw [/] saw fishies [: fish] [*] swimming in the water.
*PAR: sometimes I wonder [/] wonder where fishies [: fish] [*] go when it's cold.
*PAR: maybe they have [/] have houses under the water.
*PAR: after swimming we [//] I eat [: ate] [*] &-um ice cream with &-um chocolate things on top.
*PAR: what do you call those &-um &-um sprinkles! that's the word.
*PAR: my mom said to &-um that I could have &-um two scoops next time.
*PAR: I want to go back to the beach [/] beach next year.""",

            "School Day (Adolescent)": """*PAR: yesterday was &-um kind of a weird day at school.
*PAR: I had this big test in math and I was like really nervous about it.
*PAR: when I got there [//] when I got to class the teacher said we could use calculators.
*PAR: I was like &-oh &-um that's good because I always mess up the &-um the calculations.
*PAR: there was this one problem about &-um what do you call it &-um geometry I think.
*PAR: I couldn't remember the formula for [//] I mean I knew it but I just couldn't think of it.
*PAR: so I raised my hand and asked the teacher and she was really nice about it.
*PAR: after the test me and my friends went to lunch and we talked about how we did.
*PAR: everyone was saying it was hard but I think I did okay.
*PAR: oh and then in English class we had to read our essays out loud.
*PAR: I hate doing that because I get really nervous and I start talking fast.
*PAR: but the teacher said mine was good which made me feel better.""",

            "Adult Recovery": """*PAR: I &-um I want to talk about &-uh my &-um recovery.
*PAR: it's been &-um [//] it's hard to &-um to find the words sometimes.
*PAR: before the &-um the stroke I was &-um working at the &-uh at the bank.
*PAR: now I have to &-um practice speaking every day with my therapist.
*PAR: my wife she [//] she helps me a lot at home.
*PAR: we do &-um exercises together like &-uh reading and &-um talking about pictures.
*PAR: sometimes I get frustrated because I know what I want to say but &-um the words don't come out right.
*PAR: but I'm getting better little by little.
*PAR: the doctor says I'm making good progress.
*PAR: I hope to go back to work someday but right now I'm focusing on &-um getting better."""
        }
        
        # Event handlers
        def load_sample_transcript(sample_name):
            """Load a sample transcript"""
            return SAMPLE_TRANSCRIPTS.get(sample_name, "")
        
        def use_sample_for_analysis(sample_text, age_val, gender_val, notes):
            """Use sample transcript for analysis"""
            if not sample_text:
                return "Please load a sample transcript first."
            return analyze_transcript(sample_text, age_val, gender_val, notes)
        
        def on_analyze(transcript_text, age_val, gender_val, notes):
            """Handle analysis"""
            if not transcript_text or len(transcript_text.strip()) < 50:
                return "Error: Please provide a longer transcript for analysis."
            return analyze_transcript(transcript_text, age_val, gender_val, notes)
        
        # Connect event handlers
        load_sample_btn.click(
            load_sample_transcript,
            inputs=[sample_choice],
            outputs=[sample_transcript]
        )
        
        use_sample_btn.click(
            use_sample_for_analysis,
            inputs=[sample_transcript, age, gender, slp_notes],
            outputs=[analysis_output]
        )
        
        analyze_btn.click(
            on_analyze,
            inputs=[transcript, age, gender, slp_notes],
            outputs=[analysis_output]
        )
        
        # File upload handler
        file_upload.upload(process_upload, file_upload, transcript)
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Enhanced CASL Analysis Tool...")
    if not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not configured - analysis will show demo response")
        print("   For HuggingFace Spaces: Add ANTHROPIC_API_KEY as a secret in your space settings")
        print("   For local use: export ANTHROPIC_API_KEY='your-key-here'")
    else:
        print("✅ Claude API configured")
    
    app = create_interface()
    app.launch(show_api=False) 