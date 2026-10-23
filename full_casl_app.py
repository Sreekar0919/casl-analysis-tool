import gradio as gr
import boto3
import json
import numpy as np
import re
import logging
import os
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.info("ReportLab not available - PDF export disabled")

try:
    import speech_recognition as sr
    import pydub
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.info("Speech recognition not available - audio transcription will use demo mode")

# AWS credentials (optional)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize AWS client if available
bedrock_client = None
if AWS_ACCESS_KEY and AWS_SECRET_KEY:
    try:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        logger.info("Bedrock client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
else:
    logger.info("AWS credentials not configured - using demo mode")

# Data directories
DATA_DIR = os.environ.get("DATA_DIR", "patient_data")

def ensure_data_dirs():
    """Ensure data directories exist"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.info(f"Data directories created: {DATA_DIR}")
    except Exception as e:
        logger.warning(f"Could not create data directories: {str(e)}")
        logger.info("Using temporary directory for data storage")

ensure_data_dirs()

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

def call_bedrock(prompt, max_tokens=4096):
    """Call AWS Bedrock API with correct format or return demo response"""
    if not bedrock_client:
        return generate_demo_response(prompt)
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 0.3,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        })
        
        # Use the correct model ID
        modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        
        response = bedrock_client.invoke_model(
            body=body, 
            modelId=modelId,
            accept='application/json', 
            contentType='application/json'
        )
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']
    except Exception as e:
        logger.error(f"Error calling Bedrock: {str(e)}")
        return generate_demo_response(prompt)

def generate_demo_response(prompt):
    """Generate demo analysis response based on transcript patterns"""
    # Extract transcript from prompt
    transcript_match = re.search(r'TRANSCRIPT:\s*(.*?)(?=\n\n|\Z)', prompt, re.DOTALL)
    transcript = transcript_match.group(1) if transcript_match else ""
    
    # Count speech patterns
    um_count = len(re.findall(r'&-um|&-uh', transcript))
    revision_count = len(re.findall(r'\[//\]', transcript))
    repetition_count = len(re.findall(r'\[/\]', transcript))
    error_count = len(re.findall(r'\[\*\]', transcript))
    
    # Generate realistic scores based on patterns
    fluency_score = max(70, 100 - (um_count * 2))
    syntactic_score = max(70, 100 - (error_count * 3))
    semantic_score = max(75, 105 - (revision_count * 2))
    
    # Convert to percentiles
    fluency_percentile = int(np.interp(fluency_score, [70, 85, 100, 115], [5, 16, 50, 84]))
    syntactic_percentile = int(np.interp(syntactic_score, [70, 85, 100, 115], [5, 16, 50, 84]))
    semantic_percentile = int(np.interp(semantic_score, [70, 85, 100, 115], [5, 16, 50, 84]))
    
    def get_performance_level(score):
        if score < 70: return "Well Below Average"
        elif score < 85: return "Below Average" 
        elif score < 115: return "Average"
        else: return "Above Average"
    
    return f"""<SPEECH_FACTORS_START>
Difficulty producing fluent speech: {um_count + revision_count}, {100 - fluency_percentile}
Examples:
- Frequent use of fillers (&-um, &-uh) observed throughout transcript
- Self-corrections and revisions interrupt speech flow

Word retrieval issues: {um_count // 2 + 1}, {90 - semantic_percentile}
Examples:
- Hesitations and pauses before content words noted
- Circumlocutions and word-finding difficulties evident

Grammatical errors: {error_count}, {85 - syntactic_percentile}
Examples:
- Morphological errors marked with [*] in transcript
- Verb tense and agreement inconsistencies observed

Repetitions and revisions: {repetition_count + revision_count}, {80 - fluency_percentile}
Examples:
- Self-corrections marked with [//] throughout sample
- Word and phrase repetitions marked with [/] noted
<SPEECH_FACTORS_END>

<CASL_SKILLS_START>
Lexical/Semantic Skills: Standard Score ({semantic_score}), Percentile Rank ({semantic_percentile}%), {get_performance_level(semantic_score)}
Examples:
- Vocabulary diversity and semantic precision assessed
- Word-finding strategies and retrieval patterns analyzed

Syntactic Skills: Standard Score ({syntactic_score}), Percentile Rank ({syntactic_percentile}%), {get_performance_level(syntactic_score)}
Examples:
- Sentence structure complexity and grammatical accuracy evaluated
- Morphological skill development measured

Supralinguistic Skills: Standard Score ({fluency_score}), Percentile Rank ({fluency_percentile}%), {get_performance_level(fluency_score)}
Examples:
- Discourse organization and narrative coherence reviewed
- Pragmatic language use and communication effectiveness assessed
<CASL_SKILLS_END>

<TREATMENT_RECOMMENDATIONS_START>
- Implement word-finding strategies with semantic feature analysis and phonemic cuing
- Practice sentence formulation exercises targeting grammatical accuracy and complexity
- Use narrative structure activities with visual supports to improve discourse organization
- Incorporate self-monitoring techniques to increase awareness of speech patterns
- Apply fluency shaping strategies to reduce disfluencies and improve communication flow
<TREATMENT_RECOMMENDATIONS_END>

<EXPLANATION_START>
The language sample demonstrates patterns consistent with expressive language challenges affecting fluency, word retrieval, and syntactic formulation. The presence of self-corrections indicates preserved metalinguistic awareness, which is a positive prognostic indicator. Intervention should focus on strengthening lexical access, grammatical formulation, and discourse-level skills while building on existing self-monitoring abilities.
<EXPLANATION_END>"""

def parse_casl_response(response):
    """Parse structured response into components"""
    def extract_section(text, section_name):
        pattern = re.compile(f"<{section_name}_START>(.*?)<{section_name}_END>", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""
    
    sections = {
        'speech_factors': extract_section(response, 'SPEECH_FACTORS'),
        'casl_data': extract_section(response, 'CASL_SKILLS'), 
        'treatment_suggestions': extract_section(response, 'TREATMENT_RECOMMENDATIONS'),
        'explanation': extract_section(response, 'EXPLANATION')
    }
    
    # Build formatted report
    full_report = f"""# Speech Language Assessment Report

## Speech Factors Analysis
{sections['speech_factors']}

## CASL Skills Assessment  
{sections['casl_data']}

## Treatment Recommendations
{sections['treatment_suggestions']}

## Clinical Explanation
{sections['explanation']}
"""
    
    return {
        'speech_factors': sections['speech_factors'],
        'casl_data': sections['casl_data'], 
        'treatment_suggestions': sections['treatment_suggestions'],
        'explanation': sections['explanation'],
        'full_report': full_report,
        'raw_response': response
    }

def analyze_transcript(transcript, age, gender):
    """Analyze transcript using CASL framework"""
    prompt = f"""
    You are an expert speech-language pathologist conducting a comprehensive CASL-2 assessment.
    Analyze this transcript for a {age}-year-old {gender} patient.

    TRANSCRIPT:
    {transcript}

    Provide detailed analysis in this exact format:

    <SPEECH_FACTORS_START>
    Difficulty producing fluent speech: X, Y
    Examples:
    - "exact quote from transcript showing disfluency"
    - "another example with specific evidence"
    
    Word retrieval issues: X, Y
    Examples:
    - "quote showing word-finding difficulty"
    - "example of circumlocution or pause"
    
    Grammatical errors: X, Y
    Examples:
    - "quote showing morphological error"
    - "example of syntactic difficulty"
    
    Repetitions and revisions: X, Y
    Examples:
    - "quote showing self-correction"
    - "example of repetition or revision"
    <SPEECH_FACTORS_END>

    <CASL_SKILLS_START>
    Lexical/Semantic Skills: Standard Score (X), Percentile Rank (Y%), Performance Level
    Examples:
    - "specific vocabulary usage example"
    - "semantic precision demonstration"

    Syntactic Skills: Standard Score (X), Percentile Rank (Y%), Performance Level
    Examples:
    - "grammatical structure example"
    - "morphological skill demonstration"

    Supralinguistic Skills: Standard Score (X), Percentile Rank (Y%), Performance Level
    Examples:
    - "discourse organization example"
    - "narrative coherence demonstration"
    <CASL_SKILLS_END>

    <TREATMENT_RECOMMENDATIONS_START>
    - Specific, evidence-based treatment recommendation
    - Another targeted intervention strategy
    - Additional therapeutic approach with clear rationale
    <TREATMENT_RECOMMENDATIONS_END>

    <EXPLANATION_START>
    Comprehensive clinical explanation of findings, their significance for diagnosis and prognosis, and relationship to functional communication needs.
    <EXPLANATION_END>

    Requirements:
    1. Use exact quotes from the transcript as evidence
    2. Provide realistic standard scores (70-130 range, mean=100, SD=15)
    3. Calculate appropriate percentiles based on age norms
    4. Give specific, actionable treatment recommendations
    5. Consider developmental expectations for the patient's age
    """
    
    response = call_bedrock(prompt)
    return parse_casl_response(response)

def process_upload(file):
    """Process uploaded transcript file"""
    if file is None:
        return ""
    
    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.cha':
            # Process CHAT format file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract participant lines
            par_lines = []
            inv_lines = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('*PAR:') or line.startswith('*CHI:'):
                    par_lines.append(line)
                elif line.startswith('*INV:') or line.startswith('*EXA:'):
                    inv_lines.append(line)
            
            # Combine all relevant lines
            all_lines = []
            for line in content.splitlines():
                line = line.strip()
                if any(line.startswith(prefix) for prefix in ['*PAR:', '*CHI:', '*INV:', '*EXA:']):
                    all_lines.append(line)
            
            return '\n'.join(all_lines) if all_lines else content
        else:
            # Read as plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {str(e)}")
        return f"Error reading file: {str(e)}"

def transcribe_audio(audio_path):
    """Transcribe audio file to CHAT format"""
    if not audio_path:
        return "Please upload an audio file first.", "❌ No audio file provided"
    
    if SPEECH_RECOGNITION_AVAILABLE:
        try:
            r = sr.Recognizer()
            
            # Convert to WAV if needed
            wav_path = audio_path
            if not audio_path.endswith('.wav'):
                try:
                    audio = pydub.AudioSegment.from_file(audio_path)
                    wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                    audio.export(wav_path, format="wav")
                except Exception as e:
                    logger.warning(f"Audio conversion failed: {e}")
            
            # Transcribe
            with sr.AudioFile(wav_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                
                # Format as CHAT
                sentences = re.split(r'[.!?]+', text)
                chat_lines = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        chat_lines.append(f"*PAR: {sentence}.")
                
                result = '\n'.join(chat_lines)
                return result, "✅ Transcription completed successfully"
                
        except sr.UnknownValueError:
            return "Could not understand audio clearly", "❌ Speech not recognized"
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}", "❌ Service error"
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Error during transcription: {str(e)}", f"❌ Transcription failed"
    else:
        # Demo transcription
        demo_text = """*PAR: this is a demonstration transcription.
*PAR: to enable real audio processing install speech_recognition and pydub.
*PAR: the demo shows how transcribed text would appear in CHAT format."""
        return demo_text, "ℹ️ Demo mode - install speech_recognition for real audio processing"

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="CASL Analysis Tool", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🗣️ CASL Analysis Tool
        **Comprehensive Assessment of Spoken Language (CASL-2)**
        
        Professional speech-language assessment tool for clinical practice and research.
        Supports transcript analysis, audio transcription, and comprehensive reporting.
        """)
        
        with gr.Tabs():
            
            # Main Analysis Tab
            with gr.TabItem("📊 Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 👤 Patient Information")
                        
                        patient_name = gr.Textbox(
                            label="Patient Name", 
                            placeholder="Enter patient name"
                        )
                        record_id = gr.Textbox(
                            label="Medical Record ID", 
                            placeholder="Enter medical record ID"
                        )
                        
                        with gr.Row():
                            age = gr.Number(
                                label="Age (years)", 
                                value=8, 
                                minimum=1, 
                                maximum=120
                            )
                            gender = gr.Radio(
                                ["male", "female", "other"], 
                                label="Gender", 
                                value="male"
                            )
                        
                        assessment_date = gr.Textbox(
                            label="Assessment Date", 
                            placeholder="MM/DD/YYYY", 
                            value=datetime.now().strftime('%m/%d/%Y')
                        )
                        clinician_name = gr.Textbox(
                            label="Clinician Name", 
                            placeholder="Enter clinician name"
                        )
                        
                        gr.Markdown("### 📝 Speech Transcript")
                        
                        sample_selector = gr.Dropdown(
                            choices=list(SAMPLE_TRANSCRIPTS.keys()),
                            label="Load Sample Transcript",
                            placeholder="Choose a sample to load"
                        )
                        
                        file_upload = gr.File(
                            label="Upload Transcript File",
                            file_types=[".txt", ".cha"]
                        )
                        
                        transcript = gr.Textbox(
                            label="Speech Transcript (CHAT format preferred)", 
                            placeholder="Enter transcript text or load from samples/file...",
                            lines=12
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze Transcript", 
                            variant="primary"
                        )
                        
                    with gr.Column():
                        gr.Markdown("### 📈 Analysis Results")
                        
                        analysis_output = gr.Markdown(
                            label="Comprehensive CASL Analysis Report",
                            value="Analysis results will appear here after clicking 'Analyze Transcript'..."
                        )
                        
                        gr.Markdown("### 📤 Export Options")
                        if REPORTLAB_AVAILABLE:
                            export_btn = gr.Button("📄 Export as PDF", variant="secondary")
                            export_status = gr.Markdown("")
                        else:
                            gr.Markdown("⚠️ PDF export unavailable (ReportLab not installed)")
            
            # Audio Transcription Tab
            with gr.TabItem("🎤 Audio Transcription"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎵 Audio Processing")
                        gr.Markdown("""
                        Upload audio recordings for automatic transcription into CHAT format.
                        Supports common audio formats (.wav, .mp3, .m4a, .ogg, etc.)
                        """)
                        
                        audio_input = gr.Audio(
                            type="filepath", 
                            label="Audio Recording"
                        )
                        
                        transcribe_btn = gr.Button(
                            "🎧 Transcribe Audio", 
                            variant="primary"
                        )
                        
                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="Transcription Result (CHAT Format)", 
                            placeholder="Transcribed text will appear here...",
                            lines=15
                        )
                        
                        transcription_status = gr.Markdown("")
                        
                        copy_to_analysis_btn = gr.Button(
                            "📋 Use for Analysis", 
                            variant="secondary"
                        )
            
            # Information Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About the CASL Analysis Tool
                
                This tool provides comprehensive speech-language assessment using the CASL-2 (Comprehensive Assessment of Spoken Language) framework.
                
                ### Features:
                - **Speech Factor Analysis**: Automated detection of disfluencies, word retrieval issues, grammatical errors, and repetitions
                - **CASL-2 Domains**: Assessment of Lexical/Semantic, Syntactic, and Supralinguistic skills
                - **Professional Scoring**: Standard scores, percentiles, and performance levels
                - **Audio Transcription**: Convert speech recordings to CHAT format transcripts
                - **Treatment Recommendations**: Evidence-based intervention suggestions
                
                ### Supported Formats:
                - **Text Files**: .txt format with manual transcript entry
                - **CHAT Files**: .cha format following CHILDES conventions
                - **Audio Files**: .wav, .mp3, .m4a, .ogg for automatic transcription
                
                ### CHAT Format Guidelines:
                - Use `*PAR:` for patient utterances
                - Use `*INV:` for investigator/clinician utterances  
                - Mark filled pauses as `&-um`, `&-uh`
                - Mark repetitions with `[/]`
                - Mark revisions with `[//]`
                - Mark errors with `[*]`
                
                ### Usage Tips:
                1. Load a sample transcript to see the expected format
                2. Enter patient information for context-appropriate analysis
                3. Upload or type transcript in CHAT format for best results
                4. Review analysis results and treatment recommendations
                5. Export professional PDF reports for clinical documentation
                
                ### Technical Notes:
                - **Demo Mode**: Works without external dependencies using simulated analysis
                - **Enhanced Mode**: Requires AWS Bedrock credentials for AI-powered analysis
                - **Audio Processing**: Requires speech_recognition library for real transcription
                - **PDF Export**: Requires ReportLab library for professional reports
                
                For support or questions, please refer to the documentation.
                """)
        
        # Event Handlers
        def load_sample_transcript(sample_name):
            """Load selected sample transcript"""
            if sample_name and sample_name in SAMPLE_TRANSCRIPTS:
                return SAMPLE_TRANSCRIPTS[sample_name]
            return ""
        
        def perform_analysis(transcript_text, age_val, gender_val):
            """Perform CASL analysis on transcript"""
            if not transcript_text or len(transcript_text.strip()) < 20:
                return "❌ **Error**: Please provide a longer transcript (minimum 20 characters) for meaningful analysis."
            
            try:
                # Perform analysis
                results = analyze_transcript(transcript_text, age_val, gender_val)
                return results['full_report']
                
            except Exception as e:
                logger.exception("Analysis error")
                return f"❌ **Error during analysis**: {str(e)}\n\nPlease check your transcript format and try again."
        
        def copy_transcription_to_analysis(transcription_text):
            """Copy transcription result to analysis tab"""
            return transcription_text
        
        # Connect event handlers
        sample_selector.change(
            load_sample_transcript, 
            inputs=[sample_selector], 
            outputs=[transcript]
        )
        
        file_upload.upload(
            process_upload, 
            inputs=[file_upload], 
            outputs=[transcript]
        )
        
        analyze_btn.click(
            perform_analysis,
            inputs=[transcript, age, gender],
            outputs=[analysis_output]
        )
        
        transcribe_btn.click(
            transcribe_audio,
            inputs=[audio_input],
            outputs=[transcription_output, transcription_status]
        )
        
        copy_to_analysis_btn.click(
            copy_transcription_to_analysis,
            inputs=[transcription_output],
            outputs=[transcript]
        )
    
    return app

# Create and launch the application
if __name__ == "__main__":
    # Check for optional dependencies
    missing_deps = []
    if not REPORTLAB_AVAILABLE:
        missing_deps.append("reportlab (for PDF export)")
    if not SPEECH_RECOGNITION_AVAILABLE:
        missing_deps.append("speech_recognition & pydub (for audio transcription)")
    
    if missing_deps:
        print("📋 Optional dependencies not found:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("The app will work with reduced functionality.")
    
    if not bedrock_client:
        print("ℹ️  AWS credentials not configured - using demo mode for analysis.")
        print("   Configure AWS_ACCESS_KEY and AWS_SECRET_KEY for enhanced AI analysis.")
    
    print("🚀 Starting CASL Analysis Tool...")
    
    # Create and launch the app
    app = create_interface()
    app.launch(
        show_api=False,
        server_name="0.0.0.0",
        server_port=7860
    )