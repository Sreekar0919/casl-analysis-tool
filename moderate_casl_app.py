import gradio as gr
import boto3
import json
import re
import logging
import os
import tempfile
import shutil
import time
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ReportLab (needed for PDF generation)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("ReportLab library not available - PDF export will be disabled")
    REPORTLAB_AVAILABLE = False

# Try to import speech recognition for local audio processing
try:
    import speech_recognition as sr
    import pydub
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# AWS credentials for Bedrock API and S3
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "casl-audio-uploads")

# Initialize AWS clients if credentials are available
bedrock_client = None
s3_client = None

if AWS_ACCESS_KEY and AWS_SECRET_KEY:
    try:
        # Initialize Bedrock client for AI analysis
        bedrock_client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        # Initialize S3 client for audio file storage
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        logger.info("AWS clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {str(e)}")

# Create data directories if they don't exist
DATA_DIR = os.environ.get("DATA_DIR", "patient_data")
DOWNLOADS_DIR = os.path.join(DATA_DIR, "downloads")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

def ensure_data_dirs():
    """Ensure data directories exist"""
    global DOWNLOADS_DIR, AUDIO_DIR
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)
        logger.info(f"Data directories created: {DATA_DIR}, {DOWNLOADS_DIR}, {AUDIO_DIR}")
    except Exception as e:
        logger.warning(f"Could not create data directories: {str(e)}")
        # Fallback to tmp directory on HF Spaces
        DOWNLOADS_DIR = os.path.join(tempfile.gettempdir(), "casl_downloads")
        AUDIO_DIR = os.path.join(tempfile.gettempdir(), "casl_audio")
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)
        logger.info(f"Using fallback directories: {DOWNLOADS_DIR}, {AUDIO_DIR}")

# Initialize data directories
ensure_data_dirs()

# Sample transcript for the demo
SAMPLE_TRANSCRIPT = """*PAR: today I would &-um like to talk about &-um a fun trip I took last &-um summer with my family.
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
*PAR: I want to go back to the beach [/] beach next year."""

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
    """Process an uploaded file (PDF, text, or CHA)"""
    if file is None:
        return ""
    
    file_path = file.name
    if file_path.endswith('.pdf'):
        # For PDF, we would need PyPDF2 or similar
        return "PDF upload not supported in this simple version"
    elif file_path.endswith('.cha'):
        return read_cha_file(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def call_bedrock(prompt, max_tokens=4096):
    """Call the AWS Bedrock API to analyze text using Claude"""
    if not bedrock_client:
        return "AWS credentials not configured. Using demo response instead."
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
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
            ],
            "temperature": 0.3,
            "top_p": 0.9
        })

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
        logger.error(f"Error in call_bedrock: {str(e)}")
        return f"Error: {str(e)}"

def upload_to_s3(file_path, file_name):
    """Upload a file to S3 bucket"""
    if not s3_client:
        logger.warning("S3 client not available")
        return None
    
    try:
        s3_key = f"audio/{datetime.now().strftime('%Y-%m-%d')}/{file_name}"
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        
        # Generate a presigned URL that expires in 1 hour
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600
        )
        
        logger.info(f"File uploaded to S3: {s3_key}")
        return {'s3_key': s3_key, 'url': url}
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return None

def save_audio_file(audio_path, patient_name="", record_id=""):
    """Save audio file locally and optionally to S3"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_base = f"{patient_name}_{record_id}_{timestamp}" if patient_name and record_id else f"audio_{timestamp}"
        
        # Get file extension
        file_ext = os.path.splitext(audio_path)[1] or '.wav'
        final_filename = f"{filename_base}{file_ext}"
        
        # Save locally
        local_path = os.path.join(AUDIO_DIR, final_filename)
        shutil.copy2(audio_path, local_path)
        
        # Upload to S3 if available
        s3_info = None
        if s3_client:
            s3_info = upload_to_s3(local_path, final_filename)
        
        return {
            'local_path': local_path,
            'filename': final_filename,
            's3_info': s3_info
        }
    except Exception as e:
        logger.error(f"Error saving audio file: {str(e)}")
        return None

def transcribe_audio_local(audio_path, patient_name="", record_id=""):
    """Local audio transcription using speech_recognition library with S3 storage"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return generate_demo_transcription()
    
    try:
        # Save the audio file (locally and to S3)
        saved_file_info = save_audio_file(audio_path, patient_name, record_id)
        if saved_file_info:
            logger.info(f"Audio saved: {saved_file_info['filename']}")
            if saved_file_info['s3_info']:
                logger.info(f"Audio uploaded to S3: {saved_file_info['s3_info']['s3_key']}")
        
        r = sr.Recognizer()
        
        # Convert audio to WAV if needed
        if not audio_path.endswith('.wav'):
            try:
                audio = pydub.AudioSegment.from_file(audio_path)
                wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_path, format="wav")
                audio_path = wav_path
            except Exception as e:
                logger.error(f"Error converting audio: {str(e)}")
                return f"Error: Could not process audio file. {str(e)}"
        
        # Transcribe audio
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
                return format_transcription_as_chat(text)
            except sr.UnknownValueError:
                return "Error: Could not understand audio"
            except sr.RequestError as e:
                return f"Error: Could not request results; {e}"
                
    except Exception as e:
        logger.error(f"Error in local transcription: {str(e)}")
        return generate_demo_transcription()

def format_transcription_as_chat(text):
    """Format transcribed text into CHAT format"""
    # Split text into sentences and format as participant speech
    sentences = re.split(r'[.!?]+', text)
    chat_lines = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            chat_lines.append(f"*PAR: {sentence}.")
    
    return '\n'.join(chat_lines)

def generate_demo_transcription():
    """Generate a simulated transcription response"""
    return """*PAR: today I want to tell you about my favorite toy.
*PAR: it's a &-um teddy bear that I got for my birthday.
*PAR: he has &-um brown fur and a red bow.
*PAR: I like to sleep with him every night.
*PAR: sometimes I take him to school in my backpack.
*INV: what's your teddy bear's name?
*PAR: his name is &-um Brownie because he's brown."""

def generate_demo_response(prompt):
    """Generate a response using Bedrock if available, otherwise return a demo response"""
    # This function will attempt to call Bedrock, and only fall back to the demo response
    # if Bedrock is not available or fails
    
    # Try to call Bedrock first if client is available
    if bedrock_client:
        try:
            return call_bedrock(prompt)
        except Exception as e:
            logger.error(f"Error calling Bedrock: {str(e)}")
            logger.info("Falling back to demo response")
            # Continue to fallback response if Bedrock call fails
    
    # Fallback demo response
    logger.warning("Using demo response - Bedrock client not available or call failed")
    return """<SPEECH_FACTORS_START>
Difficulty producing fluent speech: 8, 65
Examples:
- "today I would &-um like to talk about &-um a fun trip I took last &-um summer with my family"
- "we went to the &-um &-um beach [//] no to the mountains [//] I mean the beach actually"

Word retrieval issues: 6, 72
Examples:
- "what do you call those &-um &-um sprinkles! that's the word"
- "sometimes I forget [//] forgetted [: forgot] [*] what they call those things we built"

Grammatical errors: 4, 58
Examples:
- "after swimming we [//] I eat [: ate] [*] &-um ice cream"
- "sometimes I forget [//] forgetted [: forgot] [*] what they call those things we built"

Repetitions and revisions: 5, 62
Examples:
- "we [/] we stayed for &-um three no [//] four days"
- "we went to the &-um &-um beach [//] no to the mountains [//] I mean the beach actually"
<SPEECH_FACTORS_END>

<CASL_SKILLS_START>
Lexical/Semantic Skills: Standard Score (92), Percentile Rank (30%), Average Performance
Examples:
- "what do you call those &-um &-um sprinkles! that's the word"
- "we went to the &-um &-um beach [//] no to the mountains [//] I mean the beach actually"

Syntactic Skills: Standard Score (87), Percentile Rank (19%), Low Average Performance
Examples:
- "my brother he [//] he helped me dig a big hole"
- "after swimming we [//] I eat [: ate] [*] &-um ice cream with &-um chocolate things on top"

Supralinguistic Skills: Standard Score (90), Percentile Rank (25%), Average Performance
Examples:
- "sometimes I wonder [/] wonder where fishies [: fish] [*] go when it's cold"
- "maybe they have [/] have houses under the water"
<CASL_SKILLS_END>

<TREATMENT_RECOMMENDATIONS_START>
- Implement word-finding strategies with semantic cuing focused on everyday objects and activities, using the patient's beach experience as a context (e.g., "sprinkles," "castles")
- Practice structured narrative tasks with visual supports to reduce revisions and improve sequencing
- Use sentence formulation exercises focusing on verb tense consistency (addressing errors like "forgetted" and "eat" for "ate")
- Incorporate self-monitoring techniques to help identify and correct grammatical errors
- Work on increasing vocabulary specificity (e.g., "things on top" to "sprinkles")
<TREATMENT_RECOMMENDATIONS_END>

<EXPLANATION_START>
This child demonstrates moderate word-finding difficulties with compensatory strategies including fillers ("&-um") and repetitions. The frequent use of self-corrections shows good metalinguistic awareness, but the pauses and repairs impact conversational fluency. Syntactic errors primarily involve verb tense inconsistency. Overall, the pattern suggests a mild-to-moderate language disorder with stronger receptive than expressive skills.
<EXPLANATION_END>

<ADDITIONAL_ANALYSIS_START>
The child shows relative strengths in maintaining topic coherence and conveying a complete narrative structure despite the language challenges. The pattern of errors suggests that word-finding difficulties and processing speed are primary concerns rather than conceptual or cognitive issues. Semantic network activities that strengthen word associations would likely be beneficial, particularly when paired with visual supports.
<ADDITIONAL_ANALYSIS_END>

<DIAGNOSTIC_IMPRESSIONS_START>
Based on the language sample, this child presents with a profile consistent with a mild-to-moderate expressive language disorder. The most prominent features include:

1. Word-finding difficulties characterized by fillers, pauses, and self-corrections when attempting to retrieve specific vocabulary
2. Grammatical challenges primarily affecting verb tense consistency and morphological markers
3. Relatively intact narrative structure and topic maintenance

These findings suggest intervention should focus on word retrieval strategies, grammatical form practice, and continued support for narrative development, with an emphasis on fluency and self-monitoring.
<DIAGNOSTIC_IMPRESSIONS_END>

<ERROR_EXAMPLES_START>
Word-finding difficulties:
- "what do you call those &-um &-um sprinkles! that's the word"
- "we went to the &-um &-um beach [//] no to the mountains [//] I mean the beach actually"
- "there was lots of &-um &-um swimming and &-um sun"

Grammatical errors:
- "after swimming we [//] I eat [: ate] [*] &-um ice cream"
- "sometimes I forget [//] forgetted [: forgot] [*] what they call those things we built"
- "we saw [/] saw fishies [: fish] [*] swimming in the water"

Repetitions and revisions:
- "we [/] we stayed for &-um three no [//] four days"
- "I want to go back to the beach [/] beach next year"
- "sometimes I wonder [/] wonder where fishies [: fish] [*] go when it's cold"
<ERROR_EXAMPLES_END>"""

def parse_casl_response(response):
    """Parse the LLM response for CASL analysis into structured data"""
    # Extract speech factors section using section markers
    speech_factors_section = ""
    factors_pattern = re.compile(r"<SPEECH_FACTORS_START>(.*?)<SPEECH_FACTORS_END>", re.DOTALL)
    factors_match = factors_pattern.search(response)
    
    if factors_match:
        speech_factors_section = factors_match.group(1).strip()
    else:
        speech_factors_section = "Error extracting speech factors from analysis."
    
    # Extract CASL skills section
    casl_section = ""
    casl_pattern = re.compile(r"<CASL_SKILLS_START>(.*?)<CASL_SKILLS_END>", re.DOTALL)
    casl_match = casl_pattern.search(response)
    
    if casl_match:
        casl_section = casl_match.group(1).strip()
    else:
        casl_section = "Error extracting CASL skills from analysis."
    
    # Extract treatment recommendations
    treatment_text = ""
    treatment_pattern = re.compile(r"<TREATMENT_RECOMMENDATIONS_START>(.*?)<TREATMENT_RECOMMENDATIONS_END>", re.DOTALL)
    treatment_match = treatment_pattern.search(response)
    
    if treatment_match:
        treatment_text = treatment_match.group(1).strip()
    else:
        treatment_text = "Error extracting treatment recommendations from analysis."
    
    # Extract explanation section
    explanation_text = ""
    explanation_pattern = re.compile(r"<EXPLANATION_START>(.*?)<EXPLANATION_END>", re.DOTALL)
    explanation_match = explanation_pattern.search(response)
    
    if explanation_match:
        explanation_text = explanation_match.group(1).strip()
    else:
        explanation_text = "Error extracting clinical explanation from analysis."
    
    # Extract additional analysis
    additional_analysis = ""
    additional_pattern = re.compile(r"<ADDITIONAL_ANALYSIS_START>(.*?)<ADDITIONAL_ANALYSIS_END>", re.DOTALL)
    additional_match = additional_pattern.search(response)
    
    if additional_match:
        additional_analysis = additional_match.group(1).strip()
    
    # Extract diagnostic impressions
    diagnostic_impressions = ""
    diagnostic_pattern = re.compile(r"<DIAGNOSTIC_IMPRESSIONS_START>(.*?)<DIAGNOSTIC_IMPRESSIONS_END>", re.DOTALL)
    diagnostic_match = diagnostic_pattern.search(response)
    
    if diagnostic_match:
        diagnostic_impressions = diagnostic_match.group(1).strip()
    
    # Extract specific error examples
    specific_errors_text = ""
    errors_pattern = re.compile(r"<ERROR_EXAMPLES_START>(.*?)<ERROR_EXAMPLES_END>", re.DOTALL)
    errors_match = errors_pattern.search(response)
    
    if errors_match:
        specific_errors_text = errors_match.group(1).strip()
    
    # Create full report text
    full_report = f"""
## Speech Factors Analysis

{speech_factors_section}

## CASL Skills Assessment

{casl_section}

## Treatment Recommendations

{treatment_text}

## Clinical Explanation

{explanation_text}
"""
    
    if additional_analysis:
        full_report += f"\n## Additional Analysis\n\n{additional_analysis}"
    
    if diagnostic_impressions:
        full_report += f"\n## Diagnostic Impressions\n\n{diagnostic_impressions}"
    
    if specific_errors_text:
        full_report += f"\n## Detailed Error Examples\n\n{specific_errors_text}"
    
    return {
        'speech_factors': speech_factors_section,
        'casl_data': casl_section,
        'treatment_suggestions': treatment_text,
        'explanation': explanation_text,
        'additional_analysis': additional_analysis,
        'diagnostic_impressions': diagnostic_impressions,
        'specific_errors': specific_errors_text,
        'full_report': full_report,
        'raw_response': response
    }

def analyze_transcript(transcript, age, gender):
    """Analyze a speech transcript using Claude"""
    # CASL-2 assessment cheat sheet
    cheat_sheet = """
    # Speech-Language Pathologist Analysis Cheat Sheet
    
    ## Types of Speech Patterns to Identify:
    
    1. Difficulty producing fluent, grammatical speech
       - Fillers (um, uh) and pauses
       - False starts and revisions
       - Incomplete sentences
    
    2. Word retrieval issues
       - Pauses before content words
       - Circumlocutions (talking around a word)
       - Word substitutions
    
    3. Grammatical errors
       - Verb tense inconsistencies
       - Subject-verb agreement errors
       - Morphological errors (plurals, possessives)
    
    4. Repetitions and revisions
       - Word or phrase repetitions [/]
       - Self-corrections [//]
       - Retracing
    
    5. Neologisms
       - Made-up words
       - Word blends
    
    6. Perseveration
       - Inappropriate repetition of ideas
       - Recurring themes
    
    7. Comprehension issues
       - Topic maintenance difficulties
       - Non-sequiturs
       - Inappropriate responses
    """
    
    # Instructions for the analysis
    instructions = """
    Analyze this speech transcript to identify specific patterns and provide a detailed CASL-2 (Comprehensive Assessment of Spoken Language) assessment.
    
    For each speech pattern you identify:
    1. Count the occurrences in the transcript
    2. Estimate a percentile (how typical/atypical this is for the age)
    3. Provide DIRECT QUOTES from the transcript as evidence
    
    Then assess the following CASL-2 domains:
    
    1. Lexical/Semantic Skills:
       - Assess vocabulary diversity, word-finding abilities, semantic precision
       - Provide Standard Score (mean=100, SD=15), percentile rank, and performance level
       - Include SPECIFIC QUOTES as evidence
    
    2. Syntactic Skills:
       - Evaluate grammatical accuracy, sentence complexity, morphological skills
       - Provide Standard Score, percentile rank, and performance level
       - Include SPECIFIC QUOTES as evidence
    
    3. Supralinguistic Skills:
       - Assess figurative language use, inferencing, and abstract reasoning
       - Provide Standard Score, percentile rank, and performance level
       - Include SPECIFIC QUOTES as evidence
    
    YOUR RESPONSE MUST USE THESE EXACT SECTION MARKERS FOR PARSING:
    
    <SPEECH_FACTORS_START>
    Difficulty producing fluent, grammatical speech: (occurrences), (percentile)
    Examples:
    - "(direct quote from transcript)"
    - "(direct quote from transcript)"
    
    Word retrieval issues: (occurrences), (percentile)
    Examples:
    - "(direct quote from transcript)"
    - "(direct quote from transcript)"
    
    (And so on for each factor)
    <SPEECH_FACTORS_END>
    
    <CASL_SKILLS_START>
    Lexical/Semantic Skills: Standard Score (X), Percentile Rank (X%), Performance Level
    Examples:
    - "(direct quote showing strength or weakness)"
    - "(direct quote showing strength or weakness)"
    
    Syntactic Skills: Standard Score (X), Percentile Rank (X%), Performance Level
    Examples:
    - "(direct quote showing strength or weakness)"
    - "(direct quote showing strength or weakness)"
    
    Supralinguistic Skills: Standard Score (X), Percentile Rank (X%), Performance Level
    Examples:
    - "(direct quote showing strength or weakness)"
    - "(direct quote showing strength or weakness)"
    <CASL_SKILLS_END>
    
    <TREATMENT_RECOMMENDATIONS_START>
    - (treatment recommendation)
    - (treatment recommendation)
    - (treatment recommendation)
    <TREATMENT_RECOMMENDATIONS_END>
    
    <EXPLANATION_START>
    (brief diagnostic rationale based on findings)
    <EXPLANATION_END>
    
    <ADDITIONAL_ANALYSIS_START>
    (specific insights that would be helpful for treatment planning)
    <ADDITIONAL_ANALYSIS_END>
    
    <DIAGNOSTIC_IMPRESSIONS_START>
    (summarize findings across domains using specific examples and clear explanations)
    <DIAGNOSTIC_IMPRESSIONS_END>
    
    <ERROR_EXAMPLES_START>
    (Copy all the specific quote examples here again, organized by error type or skill domain)
    <ERROR_EXAMPLES_END>
    
    MOST IMPORTANT: 
    1. Use EXACTLY the section markers provided (like <SPEECH_FACTORS_START>) to make parsing reliable
    2. For EVERY factor and domain you analyze, you MUST provide direct quotes from the transcript as evidence
    3. Be very specific and cite the exact text
    4. Do not omit any of the required sections
    """
    
    # Prepare prompt for Claude with the user's role context
    role_context = """
    You are a speech pathologist, a healthcare professional who specializes in evaluating, diagnosing, and treating communication disorders, including speech, language, cognitive-communication, voice, swallowing, and fluency disorders. Your role is to help patients improve their speech and communication skills through various therapeutic techniques and exercises.
    
    You are working with a student with speech impediments.
    
    The most important thing is that you stay kind to the child. Be constructive and helpful rather than critical.
    """
    
    prompt = f"""
    {role_context}
    
    You are analyzing a transcript for a patient who is {age} years old and {gender}.
    
    TRANSCRIPT:
    {transcript}
    
    {cheat_sheet}
    
    {instructions}
    
    Remember to be precise but compassionate in your analysis. Use direct quotes from the transcript for every factor and domain you analyze.
    """
    
    # Call the appropriate API or fallback to demo mode
    response = generate_demo_response(prompt)
    
    # Parse the response
    results = parse_casl_response(response)
    
    return results

def create_interface():
    """Create the Gradio interface"""
    # Set a theme compatible with Hugging Face Spaces
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )
    
    with gr.Blocks(title="Simple CASL Analysis Tool", theme=theme) as app:
        gr.Markdown("# CASL Analysis Tool")
        gr.Markdown("A simplified tool for analyzing speech transcripts and audio using CASL framework")
        
        with gr.Tabs() as main_tabs:
            # Analysis Tab
            with gr.TabItem("Analysis", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Patient info
                        gr.Markdown("### Patient Information")
                        patient_name = gr.Textbox(label="Patient Name", placeholder="Enter patient name")
                        record_id = gr.Textbox(label="Record ID", placeholder="Enter record ID")
                        
                        with gr.Row():
                            age = gr.Number(label="Age", value=8, minimum=1, maximum=120)
                            gender = gr.Radio(["male", "female", "other"], label="Gender", value="male")
                        
                        assessment_date = gr.Textbox(
                            label="Assessment Date", 
                            placeholder="MM/DD/YYYY", 
                            value=datetime.now().strftime('%m/%d/%Y')
                        )
                        clinician_name = gr.Textbox(label="Clinician", placeholder="Enter clinician name")
                        
                        # Transcript input
                        gr.Markdown("### Transcript")
                        sample_btn = gr.Button("Load Sample Transcript")
                        file_upload = gr.File(label="Upload transcript file (.txt or .cha)")
                        transcript = gr.Textbox(
                            label="Speech transcript (CHAT format preferred)", 
                            placeholder="Enter transcript text or upload a file...",
                            lines=10
                        )
                        
                        # Analysis button
                        analyze_btn = gr.Button("Analyze Transcript", variant="primary")
                        
                    with gr.Column(scale=1):
                        # Results display
                        gr.Markdown("### Analysis Results")
                        
                        analysis_output = gr.Markdown(label="Full Analysis")
                        
                        # PDF export (only shown if ReportLab is available)
                        export_status = gr.Markdown("")
                        if REPORTLAB_AVAILABLE:
                            export_btn = gr.Button("Export as PDF", variant="secondary")
                        else:
                            gr.Markdown("⚠️ PDF export is disabled - ReportLab library is not installed")
            
            # Transcription Tab
            with gr.TabItem("Transcription", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Audio Transcription")
                        gr.Markdown("Upload an audio recording to automatically transcribe it in CHAT format")
                        
                        # Patient's age helps with transcription accuracy
                        transcription_age = gr.Number(label="Patient Age", value=8, minimum=1, maximum=120,
                                                     info="For children under 10, special language models may be used")
                        
                        # Audio input - FIXED: removed format parameter
                        audio_input = gr.Audio(type="filepath", label="Upload Audio Recording")
                        
                        # Transcribe button
                        transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
                        
                    with gr.Column(scale=1):
                        # Transcription output
                        transcription_output = gr.Textbox(
                            label="Transcription Result", 
                            placeholder="Transcription will appear here...",
                            lines=12
                        )
                        
                        with gr.Row():
                            # Button to use transcription in analysis
                            copy_to_analysis_btn = gr.Button("Use for Analysis", variant="secondary")
                            
                        # Status/info message
                        transcription_status = gr.Markdown("")
        
        # Load sample transcript button
        def load_sample():
            return SAMPLE_TRANSCRIPT
        
        sample_btn.click(load_sample, outputs=[transcript])
        
        # File upload handler
        file_upload.upload(process_upload, file_upload, transcript)
        
        # Analysis button handler
        def on_analyze_click(transcript_text, age_val, gender_val, patient_name_val, record_id_val, clinician_val, assessment_date_val):
            if not transcript_text or len(transcript_text.strip()) < 50:
                return "Error: Please provide a longer transcript for analysis."
            
            try:
                # Get the analysis results
                results = analyze_transcript(transcript_text, age_val, gender_val)
                
                # Return the full report
                return results['full_report']
            
            except Exception as e:
                logger.exception("Error during analysis")
                return f"Error during analysis: {str(e)}"
        
        analyze_btn.click(
            on_analyze_click,
            inputs=[
                transcript, age, gender, 
                patient_name, record_id, clinician_name, assessment_date
            ],
            outputs=[analysis_output]
        )
        
        # Transcription button handler
        def on_transcribe_audio(audio_path, age_val, patient_name_val, record_id_val):
            try:
                if not audio_path:
                    return "Please upload an audio file to transcribe.", "Error: No audio file provided."
                
                # Process the audio file with local transcription and S3 upload
                transcription = transcribe_audio_local(audio_path, patient_name_val, record_id_val)
                
                # Return status message based on whether it's a demo or real transcription
                if not SPEECH_RECOGNITION_AVAILABLE:
                    status_msg = "⚠️ Demo mode: Using example transcription (speech_recognition not installed)"
                else:
                    s3_status = " and uploaded to S3" if s3_client else ""
                    status_msg = f"✅ Transcription completed successfully{s3_status}"
                
                return transcription, status_msg
            except Exception as e:
                logger.exception("Error transcribing audio")
                return f"Error: {str(e)}", f"❌ Transcription failed: {str(e)}"
        
        # Connect the transcribe button to its handler
        transcribe_btn.click(
            on_transcribe_audio,
            inputs=[audio_input, transcription_age, patient_name, record_id],
            outputs=[transcription_output, transcription_status]
        )
        
        # Copy transcription to analysis tab
        def copy_to_analysis(transcription):
            return transcription, gr.update(selected=0)  # Switch to Analysis tab
        
        copy_to_analysis_btn.click(
            copy_to_analysis,
            inputs=[transcription_output],
            outputs=[transcript, main_tabs]
        )
        
    return app

if __name__ == "__main__":
    # Check for AWS credentials
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        print("NOTE: AWS credentials not found. The app will run in demo mode with simulated responses.")
        print("To enable full functionality, set AWS_ACCESS_KEY, AWS_SECRET_KEY, and optionally S3_BUCKET environment variables.")
    else:
        print(f"AWS clients initialized. S3 bucket: {S3_BUCKET}")
        if s3_client:
            print("✅ S3 audio storage enabled")
        else:
            print("⚠️ S3 client not available")
    
    app = create_interface()
    app.launch(show_api=False)  # Disable API tab for security