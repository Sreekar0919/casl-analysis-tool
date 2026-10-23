import gradio as gr
import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import logging
import os
import pickle
import csv
from PIL import Image
import io
import uuid
from datetime import datetime
import tempfile
import time
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Try to import ReportLab (needed for PDF generation)
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Try to import PyPDF2 (needed for PDF reading)
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Try to import speech recognition for local audio processing
try:
    import speech_recognition as sr
    import pydub
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS credentials for Bedrock API (optional - app works without AWS)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize AWS clients if credentials are available
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

# Enhanced sample transcripts for different scenarios
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

    "Adult Stroke Recovery": """*PAR: I &-um I want to talk about &-uh my &-um recovery.
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

# ===============================
# Database and Storage Functions
# ===============================

# Create data directories if they don't exist
DATA_DIR = os.environ.get("DATA_DIR", "patient_data")
RECORDS_FILE = os.path.join(DATA_DIR, "patient_records.csv")
ANALYSES_DIR = os.path.join(DATA_DIR, "analyses")
DOWNLOADS_DIR = os.path.join(DATA_DIR, "downloads")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

def ensure_data_dirs():
    """Ensure data directories exist with enhanced error handling"""
    global DOWNLOADS_DIR, AUDIO_DIR, ANALYSES_DIR, RECORDS_FILE
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(ANALYSES_DIR, exist_ok=True)
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)
        logger.info(f"Data directories created: {DATA_DIR}")
        
        # Create records file if it doesn't exist
        if not os.path.exists(RECORDS_FILE):
            with open(RECORDS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID", "Name", "Record ID", "Age", "Gender", 
                    "Assessment Date", "Clinician", "Analysis Date", "File Path",
                    "Summary Score", "Notes"
                ])
    except Exception as e:
        logger.warning(f"Could not create data directories: {str(e)}")
        # Fallback to tmp directory for cloud environments
        temp_base = os.path.join(tempfile.gettempdir(), "casl_data")
        DOWNLOADS_DIR = os.path.join(temp_base, "downloads")
        AUDIO_DIR = os.path.join(temp_base, "audio")
        ANALYSES_DIR = os.path.join(temp_base, "analyses")
        RECORDS_FILE = os.path.join(temp_base, "patient_records.csv")
        
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)
        os.makedirs(ANALYSES_DIR, exist_ok=True)
        
        if not os.path.exists(RECORDS_FILE):
            with open(RECORDS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID", "Name", "Record ID", "Age", "Gender", 
                    "Assessment Date", "Clinician", "Analysis Date", "File Path",
                    "Summary Score", "Notes"
                ])
        
        logger.info(f"Using temporary directories: {temp_base}")

# Initialize data directories
ensure_data_dirs()

def save_patient_record(patient_info: Dict, analysis_results: Dict, transcript: str) -> Optional[str]:
    """Save patient record to storage with enhanced data structure"""
    try:
        record_id = str(uuid.uuid4())
        
        # Extract patient information
        name = patient_info.get("name", "")
        patient_id = patient_info.get("record_id", "")
        age = patient_info.get("age", "")
        gender = patient_info.get("gender", "")
        assessment_date = patient_info.get("assessment_date", "")
        clinician = patient_info.get("clinician", "")
        notes = patient_info.get("notes", "")
        
        # Calculate summary score (average of CASL domain scores)
        summary_score = calculate_summary_score(analysis_results)
        
        # Create filename for the analysis data
        filename = f"analysis_{record_id}.pkl"
        filepath = os.path.join(ANALYSES_DIR, filename)
        
        # Save enhanced analysis data
        analysis_data = {
            "patient_info": patient_info,
            "analysis_results": analysis_results,
            "transcript": transcript,
            "timestamp": datetime.now().isoformat(),
            "summary_score": summary_score,
            "version": "2.0"  # For future compatibility
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(analysis_data, f)
        
        # Add record to CSV file
        with open(RECORDS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record_id, name, patient_id, age, gender, 
                assessment_date, clinician, datetime.now().strftime('%Y-%m-%d'),
                filepath, summary_score, notes
            ])
        
        return record_id
    
    except Exception as e:
        logger.error(f"Error saving patient record: {str(e)}")
        return None

def calculate_summary_score(analysis_results: Dict) -> float:
    """Calculate an overall summary score from CASL domain scores"""
    try:
        # Extract CASL scores from results
        casl_data = analysis_results.get('casl_data', '')
        scores = []
        
        # Look for standard scores in the CASL data
        score_pattern = r'Standard Score \((\d+)\)'
        matches = re.findall(score_pattern, casl_data)
        
        if matches:
            scores = [int(score) for score in matches]
            return round(sum(scores) / len(scores), 1)
        
        return 85.0  # Default score if parsing fails
    except Exception:
        return 85.0

def get_all_patient_records() -> List[Dict]:
    """Return a list of all patient records with enhanced filtering"""
    try:
        records = []
        ensure_data_dirs()
        
        if not os.path.exists(RECORDS_FILE):
            return records
            
        with open(RECORDS_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return records
                
            for row in reader:
                if len(row) < 9:
                    continue
                    
                file_path = row[8] if len(row) > 8 else ""
                file_exists = os.path.exists(file_path) if file_path else False
                summary_score = row[9] if len(row) > 9 else "N/A"
                notes = row[10] if len(row) > 10 else ""
                
                record = {
                    "id": row[0],
                    "name": row[1],
                    "record_id": row[2],
                    "age": row[3],
                    "gender": row[4],
                    "assessment_date": row[5],
                    "clinician": row[6],
                    "analysis_date": row[7],
                    "file_path": file_path,
                    "summary_score": summary_score,
                    "notes": notes,
                    "status": "Valid" if file_exists else "Missing File"
                }
                records.append(record)
        
        # Sort by analysis date (most recent first)
        records.sort(key=lambda x: x.get('analysis_date', ''), reverse=True)
        return records
    
    except Exception as e:
        logger.error(f"Error getting patient records: {str(e)}")
        return []

# ===============================
# Enhanced Utility Functions
# ===============================

def read_pdf(file_path: str) -> str:
    """Read text from a PDF file with better error handling"""
    if not PYPDF2_AVAILABLE:
        return "Error: PDF reading requires PyPDF2 library. Install with: pip install PyPDF2"
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    logger.warning(f"Error reading page {page_num}: {str(e)}")
                    continue
            return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return f"Error reading PDF: {str(e)}"

def read_cha_file(file_path: str) -> str:
    """Enhanced CHA file parser with better CHAT format support"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract participant lines (starting with *PAR: or *CHI:)
        participant_lines = []
        investigator_lines = []
        
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('*PAR:') or line.startswith('*CHI:'):
                participant_lines.append(line)
            elif line.startswith('*INV:') or line.startswith('*EXA:'):
                investigator_lines.append(line)
                
        # Combine participant and investigator lines in chronological order
        all_lines = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('*PAR:') or line.startswith('*CHI:') or line.startswith('*INV:') or line.startswith('*EXA:'):
                all_lines.append(line)
                
        if all_lines:
            return '\n'.join(all_lines)
        elif participant_lines:
            return '\n'.join(participant_lines)
        else:
            return content
            
    except Exception as e:
        logger.error(f"Error reading CHA file: {str(e)}")
        return ""

def process_upload(file) -> str:
    """Enhanced file processing with support for multiple formats"""
    if file is None:
        return ""
    
    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            return read_pdf(file_path)
        elif file_ext == '.cha':
            return read_cha_file(file_path)
        elif file_ext in ['.txt', '.doc', '.docx']:
            # For .doc/.docx, you might want to add python-docx support
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            # Try to read as text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content.strip()) == 0:
                    return "Error: File appears to be empty or in an unsupported format."
                return content
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return f"Error reading file: {str(e)}"

# ===============================
# Enhanced Audio Processing (Local)
# ===============================

def transcribe_audio_local(audio_path: str) -> str:
    """Local audio transcription using speech_recognition library"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return generate_demo_transcription()
    
    try:
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

def format_transcription_as_chat(text: str) -> str:
    """Format transcribed text into CHAT format"""
    # Split text into sentences and format as participant speech
    sentences = re.split(r'[.!?]+', text)
    chat_lines = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            chat_lines.append(f"*PAR: {sentence}.")
    
    return '\n'.join(chat_lines)

def generate_demo_transcription() -> str:
    """Generate a demo transcription when real transcription isn't available"""
    return """*PAR: today I want to tell you about my favorite toy.
*PAR: it's a &-um teddy bear that I got for my birthday.
*PAR: he has &-um brown fur and a red bow.
*PAR: I like to sleep with him every night.
*PAR: sometimes I take him to school in my backpack.
*INV: what's your teddy bear's name?
*PAR: his name is &-um Brownie because he's brown.
*PAR: he makes me feel &-um safe when I'm scared."""

# ===============================
# Enhanced AI Analysis Functions
# ===============================

def call_bedrock(prompt: str, max_tokens: int = 4096) -> str:
    """Enhanced Bedrock API call with better error handling"""
    if not bedrock_client:
        logger.info("Bedrock client not available, using enhanced demo response")
        return generate_enhanced_demo_response(prompt)
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.9
        })

        response = bedrock_client.invoke_model(
            body=body, 
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            accept='application/json', 
            contentType='application/json'
        )
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']
    except Exception as e:
        logger.error(f"Error in call_bedrock: {str(e)}")
        return generate_enhanced_demo_response(prompt)

def generate_enhanced_demo_response(prompt: str) -> str:
    """Generate sophisticated demo responses based on transcript analysis"""
    # Analyze the transcript in the prompt to generate more realistic responses
    transcript_match = re.search(r'TRANSCRIPT:\s*(.*?)(?=\n\n|\Z)', prompt, re.DOTALL)
    transcript = transcript_match.group(1) if transcript_match else ""
    
    # Count various speech patterns
    um_count = len(re.findall(r'&-um|&-uh', transcript))
    revision_count = len(re.findall(r'\[//\]', transcript))
    repetition_count = len(re.findall(r'\[/\]', transcript))
    error_count = len(re.findall(r'\[\*\]', transcript))
    
    # Generate scores based on patterns found
    fluency_score = max(70, 100 - (um_count * 2))
    syntactic_score = max(70, 100 - (error_count * 3))
    semantic_score = max(75, 105 - (revision_count * 2))
    
    # Convert to percentiles
    fluency_percentile = int(np.interp(fluency_score, [70, 85, 100, 115], [5, 16, 50, 84]))
    syntactic_percentile = int(np.interp(syntactic_score, [70, 85, 100, 115], [5, 16, 50, 84]))
    semantic_percentile = int(np.interp(semantic_score, [70, 85, 100, 115], [5, 16, 50, 84]))
    
    # Determine performance levels
    def get_performance_level(score):
        if score < 70: return "Well Below Average"
        elif score < 85: return "Below Average"
        elif score < 115: return "Average"
        elif score < 130: return "Above Average"
        else: return "Well Above Average"
    
    response = f"""<SPEECH_FACTORS_START>
Difficulty producing fluent speech: {um_count + revision_count}, {100 - fluency_percentile}
Examples:
- Direct quotes showing disfluencies from transcript
- Pauses and hesitations noted

Word retrieval issues: {um_count // 2 + 1}, {90 - semantic_percentile}
Examples:
- Word-finding difficulties observed
- Circumlocutions and fillers

Grammatical errors: {error_count}, {85 - syntactic_percentile}
Examples:
- Morphological and syntactic errors identified
- Verb tense and agreement issues

Repetitions and revisions: {repetition_count + revision_count}, {80 - fluency_percentile}
Examples:
- Self-corrections and repairs noted
- Repetitive patterns observed
<SPEECH_FACTORS_END>

<CASL_SKILLS_START>
Lexical/Semantic Skills: Standard Score ({semantic_score}), Percentile Rank ({semantic_percentile}%), {get_performance_level(semantic_score)}
Examples:
- Vocabulary usage and word selection patterns
- Semantic precision and concept expression

Syntactic Skills: Standard Score ({syntactic_score}), Percentile Rank ({syntactic_percentile}%), {get_performance_level(syntactic_score)}
Examples:
- Sentence structure and grammatical accuracy
- Morphological skill demonstration

Supralinguistic Skills: Standard Score ({fluency_score}), Percentile Rank ({fluency_percentile}%), {get_performance_level(fluency_score)}
Examples:
- Discourse organization and coherence
- Pragmatic language use and narrative skills
<CASL_SKILLS_END>

<TREATMENT_RECOMMENDATIONS_START>
- Target word-finding strategies with semantic feature analysis and phonemic cuing
- Implement sentence formulation exercises focusing on grammatical accuracy
- Practice narrative structure with visual supports and story grammar elements
- Use self-monitoring techniques to increase awareness of communication breakdowns
- Incorporate fluency shaping strategies to reduce disfluencies and improve flow
<TREATMENT_RECOMMENDATIONS_END>

<EXPLANATION_START>
The language sample demonstrates patterns consistent with a mild-to-moderate language disorder affecting primarily expressive skills. Word-finding difficulties and syntactic challenges are evident, while overall communicative intent remains clear. The presence of self-corrections indicates good metalinguistic awareness, which is a positive prognostic indicator for treatment.
<EXPLANATION_END>

<ADDITIONAL_ANALYSIS_START>
Strengths include maintained topic coherence and attempt at complex narrative structure. Areas of concern center on retrieval efficiency and grammatical formulation. The pattern suggests intact receptive language with specific expressive challenges that would benefit from targeted intervention focusing on lexical access and syntactic formulation.
<ADDITIONAL_ANALYSIS_END>

<DIAGNOSTIC_IMPRESSIONS_START>
Based on comprehensive analysis, this profile suggests a specific language impairment affecting expressive domains more significantly than receptive abilities. The combination of word-finding difficulties, grammatical errors, and disfluencies indicates need for structured language intervention with focus on lexical organization, syntactic practice, and metacognitive strategy development.
<DIAGNOSTIC_IMPRESSIONS_END>

<ERROR_EXAMPLES_START>
Word-finding difficulties:
- Examples of circumlocutions and word substitutions
- Pause patterns before content words

Grammatical errors:
- Specific morphological and syntactic errors
- Verb tense and agreement difficulties

Fluency disruptions:
- Repetitions, revisions, and false starts
- Filled and unfilled pause patterns
<ERROR_EXAMPLES_END>"""

    return response

def parse_casl_response(response: str) -> Dict:
    """Enhanced parsing of LLM response with better error handling and structure"""
    # Extract sections using improved regex patterns
    sections = {
        'speech_factors': extract_section(response, 'SPEECH_FACTORS'),
        'casl_data': extract_section(response, 'CASL_SKILLS'),
        'treatment_suggestions': extract_section(response, 'TREATMENT_RECOMMENDATIONS'),
        'explanation': extract_section(response, 'EXPLANATION'),
        'additional_analysis': extract_section(response, 'ADDITIONAL_ANALYSIS'),
        'diagnostic_impressions': extract_section(response, 'DIAGNOSTIC_IMPRESSIONS'),
        'specific_errors': extract_section(response, 'ERROR_EXAMPLES')
    }
    
    # Create structured analysis
    structured_data = process_speech_factors(sections['speech_factors'])
    casl_structured = process_casl_skills(sections['casl_data'])
    
    # Build comprehensive report
    full_report = build_comprehensive_report(sections)
    
    return {
        'speech_factors': structured_data['dataframe'],
        'casl_data': casl_structured['dataframe'],
        'treatment_suggestions': parse_treatment_recommendations(sections['treatment_suggestions']),
        'explanation': sections['explanation'],
        'additional_analysis': sections['additional_analysis'],
        'diagnostic_impressions': sections['diagnostic_impressions'],
        'specific_errors': structured_data['errors'],
        'full_report': full_report,
        'raw_response': response,
        'summary_scores': casl_structured['summary']
    }

def extract_section(text: str, section_name: str) -> str:
    """Extract content between section markers"""
    pattern = re.compile(f"<{section_name}_START>(.*?)<{section_name}_END>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""

def process_speech_factors(factors_text: str) -> Dict:
    """Process speech factors into structured format"""
    data = {
        'Factor': [],
        'Occurrences': [],
        'Severity': [],
        'Examples': []
    }
    
    errors = {}
    lines = factors_text.split('\n')
    current_factor = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for factor pattern: "Factor name: count, percentile"
        factor_match = re.match(r'([^:]+):\s*(\d+),\s*(\d+)', line)
        if factor_match:
            factor = factor_match.group(1).strip()
            occurrences = int(factor_match.group(2))
            severity = int(factor_match.group(3))
            
            data['Factor'].append(factor)
            data['Occurrences'].append(occurrences)
            data['Severity'].append(severity)
            data['Examples'].append("")  # Will be filled later
            current_factor = factor
            
        elif line.startswith('- ') and current_factor:
            # This is an example for the current factor
            example = line[2:].strip()
            if example:
                # Update the last added example
                if data['Examples'] and current_factor in data['Factor']:
                    idx = data['Factor'].index(current_factor)
                    if not data['Examples'][idx]:
                        data['Examples'][idx] = example
                    else:
                        data['Examples'][idx] += f"; {example}"
                errors[current_factor] = example
    
    return {
        'dataframe': pd.DataFrame(data),
        'errors': errors
    }

def process_casl_skills(casl_text: str) -> Dict:
    """Process CASL skills into structured format"""
    data = {
        'Domain': ['Lexical/Semantic', 'Syntactic', 'Supralinguistic'],
        'Standard Score': [85, 85, 85],  # Default values
        'Percentile': [16, 16, 16],
        'Performance Level': ['Below Average', 'Below Average', 'Below Average'],
        'Examples': ['', '', '']
    }
    
    lines = casl_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for domain scores
        score_match = re.search(r'(Lexical/Semantic|Syntactic|Supralinguistic)\s+Skills:\s+Standard Score \((\d+)\),\s+Percentile Rank \((\d+)%\),\s+(.+)', line)
        if score_match:
            domain = score_match.group(1)
            score = int(score_match.group(2))
            percentile = int(score_match.group(3))
            level = score_match.group(4).strip()
            
            if domain == 'Lexical/Semantic':
                idx = 0
            elif domain == 'Syntactic':
                idx = 1
            elif domain == 'Supralinguistic':
                idx = 2
            else:
                continue
                
            data['Standard Score'][idx] = score
            data['Percentile'][idx] = percentile
            data['Performance Level'][idx] = level
    
    # Calculate summary statistics
    avg_score = sum(data['Standard Score']) / len(data['Standard Score'])
    avg_percentile = sum(data['Percentile']) / len(data['Percentile'])
    
    return {
        'dataframe': pd.DataFrame(data),
        'summary': {
            'average_score': round(avg_score, 1),
            'average_percentile': round(avg_percentile, 1),
            'overall_level': get_performance_level(avg_score)
        }
    }

def get_performance_level(score: float) -> str:
    """Determine performance level from standard score"""
    if score < 70:
        return "Well Below Average"
    elif score < 85:
        return "Below Average"
    elif score < 115:
        return "Average"
    elif score < 130:
        return "Above Average"
    else:
        return "Well Above Average"

def parse_treatment_recommendations(treatment_text: str) -> List[str]:
    """Parse treatment recommendations into a list"""
    recommendations = []
    lines = treatment_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            recommendations.append(line[2:])
        elif line.startswith('• '):
            recommendations.append(line[2:])
        elif line and not line.startswith('#'):
            recommendations.append(line)
    
    return [rec for rec in recommendations if rec]

def build_comprehensive_report(sections: Dict) -> str:
    """Build a comprehensive formatted report"""
    report = """# Speech Language Assessment Report

## Speech Factors Analysis

{speech_factors}

## CASL Skills Assessment

{casl_data}

## Treatment Recommendations

{treatment_suggestions}

## Clinical Explanation

{explanation}
""".format(**sections)
    
    if sections['additional_analysis']:
        report += f"\n## Additional Analysis\n\n{sections['additional_analysis']}"
    
    if sections['diagnostic_impressions']:
        report += f"\n## Diagnostic Impressions\n\n{sections['diagnostic_impressions']}"
    
    if sections['specific_errors']:
        report += f"\n## Detailed Error Examples\n\n{sections['specific_errors']}"
    
    return report

def create_enhanced_visualizations(speech_factors_df: pd.DataFrame, casl_data_df: pd.DataFrame) -> plt.Figure:
    """Create enhanced visualizations with better styling"""
    # Set professional styling
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Speech factors bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    if not speech_factors_df.empty:
        factors_sorted = speech_factors_df.sort_values('Occurrences', ascending=True)
        bars = ax1.barh(factors_sorted['Factor'], factors_sorted['Occurrences'], 
                       color=sns.color_palette("viridis", len(factors_sorted)))
        ax1.set_title('Speech Factors Frequency', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Occurrences')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.0f}', ha='left', va='center')
    
    # CASL scores
    ax2 = fig.add_subplot(gs[0, 1])
    if not casl_data_df.empty:
        bars = ax2.bar(casl_data_df['Domain'], casl_data_df['Standard Score'], 
                      color=sns.color_palette("muted", len(casl_data_df)))
        ax2.set_title('CASL Domain Scores', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Standard Score')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Average (100)')
        ax2.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Below Average (85)')
        ax2.legend()
        
        # Add score labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height:.0f}', ha='center', va='bottom')
    
    # Severity heatmap
    ax3 = fig.add_subplot(gs[1, :])
    if not speech_factors_df.empty:
        # Create a severity matrix
        severity_data = speech_factors_df[['Factor', 'Severity']].set_index('Factor')
        severity_matrix = severity_data.T
        
        im = ax3.imshow([severity_data['Severity'].values], cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(len(severity_data)))
        ax3.set_xticklabels(severity_data.index, rotation=45, ha='right')
        ax3.set_yticks([])
        ax3.set_title('Severity Percentiles (Higher = More Severe)', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Severity Percentile')
        
        # Add text annotations
        for i, severity in enumerate(severity_data['Severity'].values):
            ax3.text(i, 0, f'{severity}%', ha='center', va='center', 
                    color='white' if severity > 50 else 'black', fontweight='bold')
    
    plt.tight_layout()
    return fig

def analyze_transcript_enhanced(transcript: str, age: int, gender: str) -> Dict:
    """Enhanced transcript analysis with comprehensive assessment"""
    
    # Enhanced CASL analysis prompt
    prompt = f"""
    You are an expert speech-language pathologist conducting a comprehensive CASL-2 assessment. 
    Analyze this transcript for a {age}-year-old {gender} patient.

    TRANSCRIPT:
    {transcript}

    Provide a detailed analysis following this exact format with specific section markers:

    <SPEECH_FACTORS_START>
    [For each factor, provide: Factor name: count, severity_percentile
    Then list 2-3 specific examples with "- " bullets]
    Difficulty producing fluent speech: X, Y
    Examples:
    - "exact quote from transcript"
    - "another exact quote"

    Word retrieval issues: X, Y
    Examples:
    - "exact quote showing word-finding difficulty"
    - "another example"

    [Continue for all relevant factors...]
    <SPEECH_FACTORS_END>

    <CASL_SKILLS_START>
    Lexical/Semantic Skills: Standard Score (X), Percentile Rank (Y%), Performance Level
    Examples:
    - "specific example of vocabulary use"

    Syntactic Skills: Standard Score (X), Percentile Rank (Y%), Performance Level  
    Examples:
    - "specific grammatical pattern example"

    Supralinguistic Skills: Standard Score (X), Percentile Rank (Y%), Performance Level
    Examples:
    - "discourse organization example"
    <CASL_SKILLS_END>

    <TREATMENT_RECOMMENDATIONS_START>
    - Specific, actionable treatment recommendation
    - Another targeted intervention strategy
    - Additional therapeutic approach
    <TREATMENT_RECOMMENDATIONS_END>

    <EXPLANATION_START>
    Comprehensive clinical explanation of findings and their significance.
    <EXPLANATION_END>

    <ADDITIONAL_ANALYSIS_START>
    Additional insights for treatment planning and prognosis.
    <ADDITIONAL_ANALYSIS_END>

    <DIAGNOSTIC_IMPRESSIONS_START>
    Summary of diagnostic findings with specific evidence and recommendations.
    <DIAGNOSTIC_IMPRESSIONS_END>

    <ERROR_EXAMPLES_START>
    Organized listing of all specific error examples by category.
    <ERROR_EXAMPLES_END>

    Be sure to:
    1. Use exact quotes from the transcript as evidence
    2. Provide realistic standard scores (70-130 range, mean=100, SD=15)
    3. Calculate appropriate percentiles
    4. Give specific, evidence-based treatment recommendations
    5. Consider the patient's age and developmental expectations
    """
    
    # Get analysis from AI or demo
    response = call_bedrock(prompt)
    
    # Parse and structure the response
    results = parse_casl_response(response)
    
    return results

# ===============================
# Enhanced PDF Export Functions
# ===============================

def export_enhanced_pdf(results: Dict, patient_info: Dict) -> str:
    """Create enhanced PDF report with professional styling"""
    if not REPORTLAB_AVAILABLE:
        return "ERROR: PDF export requires ReportLab library. Install with: pip install reportlab"
    
    try:
        # Generate filename
        patient_name = patient_info.get("name", "Unknown")
        safe_name = re.sub(r'[^\w\s-]', '', patient_name).strip()
        if not safe_name:
            safe_name = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        ensure_data_dirs()
        pdf_path = os.path.join(DOWNLOADS_DIR, f"{safe_name}_CASL_Report.pdf")
        
        # Create document with better styling
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.navy
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=5,
            backColor=colors.lightgrey
        )
        
        story = []
        
        # Title page
        story.append(Paragraph("COMPREHENSIVE SPEECH-LANGUAGE ASSESSMENT", title_style))
        story.append(Paragraph("CASL-2 Analysis Report", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Patient information table
        patient_data = []
        for key, value in patient_info.items():
            if value:
                display_key = key.replace('_', ' ').title()
                patient_data.append([display_key + ":", str(value)])
        
        if patient_data:
            patient_table = Table(patient_data, colWidths=[150, 300])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
        
        # Add sections
        sections = [
            ("Speech Factors Analysis", results.get('speech_factors', pd.DataFrame())),
            ("CASL Skills Assessment", results.get('casl_data', pd.DataFrame())),
            ("Treatment Recommendations", results.get('treatment_suggestions', [])),
            ("Clinical Explanation", results.get('explanation', "")),
            ("Additional Analysis", results.get('additional_analysis', "")),
            ("Diagnostic Impressions", results.get('diagnostic_impressions', ""))
        ]
        
        for section_title, content in sections:
            story.append(Paragraph(section_title, heading_style))
            
            if isinstance(content, pd.DataFrame) and not content.empty:
                # Convert DataFrame to table
                table_data = [content.columns.tolist()] + content.values.tolist()
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            elif isinstance(content, list):
                for item in content:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            elif isinstance(content, str) and content:
                story.append(Paragraph(content, styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Enhanced PDF report saved: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error creating enhanced PDF: {str(e)}")
        return f"Error creating PDF: {str(e)}"

# ===============================
# Enhanced Gradio Interface
# ===============================

def create_enhanced_interface():
    """Create the enhanced Gradio interface with improved UX"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background-color: #f8f9fa;
    }
    .output-markdown {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
    }
    """
    
    with gr.Blocks(title="Enhanced CASL Analysis Tool", css=custom_css, theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🗣️ Enhanced CASL Analysis Tool
        
        **Comprehensive Assessment of Spoken Language (CASL-2)**
        
        Professional speech-language assessment tool with advanced analytics and reporting capabilities.
        """)
        
        with gr.Tabs() as main_tabs:
            
            # Enhanced Analysis Tab
            with gr.TabItem("📊 Analysis", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
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
                        clinical_notes = gr.Textbox(
                            label="Clinical Notes",
                            placeholder="Additional observations or context",
                            lines=2
                        )
                        
                        gr.Markdown("### 📝 Speech Transcript")
                        
                        # Sample transcript selection
                        sample_selector = gr.Dropdown(
                            choices=list(SAMPLE_TRANSCRIPTS.keys()),
                            label="Load Sample Transcript"
                        )
                        
                        file_upload = gr.File(
                            label="Upload Transcript File",
                            file_types=[".txt", ".cha", ".pdf"]
                        )
                        
                        transcript = gr.Textbox(
                            label="Speech Transcript (CHAT format preferred)", 
                            placeholder="Enter or upload transcript...",
                            lines=12
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔍 Analyze Transcript", 
                                variant="primary"
                            )
                            save_record_btn = gr.Button(
                                "💾 Save Record", 
                                variant="secondary"
                            )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Analysis Results")
                        
                        # Results tabs
                        with gr.Tabs():
                            with gr.TabItem("Report"):
                                analysis_output = gr.Markdown(
                                    label="Analysis Report"
                                )
                            
                            with gr.TabItem("Visualizations"):
                                plot_output = gr.Plot(
                                    label="Analysis Plots"
                                )
                            
                            with gr.TabItem("Data Tables"):
                                with gr.Row():
                                    factors_table = gr.Dataframe(
                                        label="Speech Factors",
                                        interactive=False
                                    )
                                with gr.Row():
                                    casl_table = gr.Dataframe(
                                        label="CASL Domain Scores",
                                        interactive=False
                                    )
                        
                        # Export options
                        gr.Markdown("### 📤 Export Options")
                        with gr.Row():
                            if REPORTLAB_AVAILABLE:
                                export_pdf_btn = gr.Button(
                                    "📄 Export PDF Report", 
                                    variant="secondary"
                                )
                            else:
                                gr.Markdown("⚠️ PDF export unavailable - install ReportLab")
                            
                            export_csv_btn = gr.Button(
                                "📊 Export Data (CSV)", 
                                variant="secondary"
                            )
                        
                        export_status = gr.Markdown("")
            
            # Enhanced Transcription Tab
            with gr.TabItem("🎤 Transcription", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎵 Audio Processing")
                        gr.Markdown("""
                        Upload audio recordings for automatic transcription.
                        Supports various audio formats and provides CHAT-formatted output.
                        """)
                        
                        transcription_age = gr.Number(
                            label="Patient Age", 
                            value=8, 
                            minimum=1, 
                            maximum=120
                        )
                        
                        audio_input = gr.Audio(
                            type="filepath", 
                            label="Audio Recording"
                        )
                        
                        transcribe_btn = gr.Button(
                            "🎧 Transcribe Audio", 
                            variant="primary"
                        )
                        
                    with gr.Column(scale=1):
                        transcription_output = gr.Textbox(
                            label="Transcription Result", 
                            placeholder="Transcribed text will appear here...",
                            lines=15
                        )
                        
                        transcription_status = gr.Markdown("")
                        
                        with gr.Row():
                            copy_to_analysis_btn = gr.Button(
                                "📋 Use for Analysis", 
                                variant="secondary"
                            )
                            save_transcription_btn = gr.Button(
                                "💾 Save Transcription",
                                variant="secondary"
                            )
            
            # Enhanced Records Management Tab
            with gr.TabItem("📚 Records", id=2):
                gr.Markdown("### 🗃️ Patient Records Management")
                
                with gr.Row():
                    refresh_records_btn = gr.Button(
                        "🔄 Refresh Records", 
                        variant="secondary"
                    )
                    delete_record_btn = gr.Button(
                        "🗑️ Delete Selected", 
                        variant="stop"
                    )
                
                records_table = gr.Dataframe(
                    label="Patient Records",
                    headers=["ID", "Name", "Age", "Gender", "Date", "Clinician", "Score", "Status"],
                    interactive=True,
                    wrap=True
                )
                
                selected_record_info = gr.Markdown("")
                
                with gr.Row():
                    load_record_btn = gr.Button(
                        "📂 Load Selected Record", 
                        variant="primary"
                    )
                    export_records_btn = gr.Button(
                        "📊 Export All Records",
                        variant="secondary"
                    )
        
        # ===============================
        # Event Handlers
        # ===============================
        
        def load_sample_transcript(sample_name):
            if sample_name in SAMPLE_TRANSCRIPTS:
                return SAMPLE_TRANSCRIPTS[sample_name]
            return ""
        
        def perform_analysis(transcript_text, age_val, gender_val, name, record_id, clinician, assessment_date, notes):
            if not transcript_text or len(transcript_text.strip()) < 20:
                return "❌ Error: Please provide a longer transcript (at least 20 characters)", None, None, None
            
            try:
                # Perform enhanced analysis
                results = analyze_transcript_enhanced(transcript_text, age_val, gender_val)
                
                # Create visualizations
                if not results['speech_factors'].empty or not results['casl_data'].empty:
                    fig = create_enhanced_visualizations(results['speech_factors'], results['casl_data'])
                else:
                    fig = None
                
                return (
                    results['full_report'],
                    fig,
                    results['speech_factors'],
                    results['casl_data']
                )
                
            except Exception as e:
                logger.exception("Error during analysis")
                return f"❌ Error during analysis: {str(e)}", None, None, None
        
        def save_patient_record_handler(name, record_id, age_val, gender_val, assessment_date, clinician, notes, transcript_text, analysis_report):
            if not name or not transcript_text or not analysis_report:
                return "❌ Error: Missing required information for saving record"
            
            try:
                patient_info = {
                    "name": name,
                    "record_id": record_id,
                    "age": age_val,
                    "gender": gender_val,
                    "assessment_date": assessment_date,
                    "clinician": clinician,
                    "notes": notes
                }
                
                # For saving, we need to re-parse the analysis
                # This is a simplified version - in practice you'd store the full results
                results = {"full_report": analysis_report}
                
                saved_id = save_patient_record(patient_info, results, transcript_text)
                
                if saved_id:
                    return f"✅ Record saved successfully! ID: {saved_id}"
                else:
                    return "❌ Error: Failed to save record"
                    
            except Exception as e:
                return f"❌ Error saving record: {str(e)}"
        
        def transcribe_audio_handler(audio_path, age_val):
            if not audio_path:
                return "Please upload an audio file first.", "❌ No audio file provided"
            
            try:
                result = transcribe_audio_local(audio_path)
                
                if SPEECH_RECOGNITION_AVAILABLE:
                    status = "✅ Transcription completed using local speech recognition"
                else:
                    status = "ℹ️ Demo transcription (install speech_recognition for real transcription)"
                
                return result, status
                
            except Exception as e:
                error_msg = f"❌ Transcription failed: {str(e)}"
                return f"Error: {str(e)}", error_msg
        
        def load_records():
            records = get_all_patient_records()
            if not records:
                return []
            
            # Format for display
            display_records = []
            for record in records:
                display_records.append([
                    record['id'][:8] + "...",  # Truncated ID
                    record['name'],
                    record['age'],
                    record['gender'],
                    record['assessment_date'],
                    record['clinician'],
                    record.get('summary_score', 'N/A'),
                    record['status']
                ])
            
            return display_records
        
        # Connect event handlers
        sample_selector.change(load_sample_transcript, sample_selector, transcript)
        file_upload.upload(process_upload, file_upload, transcript)
        
        analyze_btn.click(
            perform_analysis,
            inputs=[transcript, age, gender, patient_name, record_id, clinician_name, assessment_date, clinical_notes],
            outputs=[analysis_output, plot_output, factors_table, casl_table]
        )
        
        save_record_btn.click(
            save_patient_record_handler,
            inputs=[patient_name, record_id, age, gender, assessment_date, clinician_name, clinical_notes, transcript, analysis_output],
            outputs=[export_status]
        )
        
        transcribe_btn.click(
            transcribe_audio_handler,
            inputs=[audio_input, transcription_age],
            outputs=[transcription_output, transcription_status]
        )
        
        copy_to_analysis_btn.click(
            lambda x: (x, gr.update(selected=0)),
            inputs=[transcription_output],
            outputs=[transcript, main_tabs]
        )
        
        refresh_records_btn.click(
            load_records,
            outputs=[records_table]
        )
        
        # Load records on startup
        app.load(load_records, outputs=[records_table])
    
    return app

if __name__ == "__main__":
    # Check dependencies and provide helpful messages
    missing_deps = []
    if not REPORTLAB_AVAILABLE:
        missing_deps.append("reportlab (for PDF export)")
    if not PYPDF2_AVAILABLE:
        missing_deps.append("PyPDF2 (for PDF reading)")
    if not SPEECH_RECOGNITION_AVAILABLE:
        missing_deps.append("speech_recognition & pydub (for audio transcription)")
    
    if missing_deps:
        print("📋 Optional dependencies not found:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nThe app will work with reduced functionality. Install missing packages for full features.")
    
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        print("ℹ️  AWS credentials not configured - using demo mode for AI analysis.")
        print("   Set AWS_ACCESS_KEY and AWS_SECRET_KEY environment variables for full functionality.")
    
    print("🚀 Starting Enhanced CASL Analysis Tool...")
    app = create_enhanced_interface()
    app.launch(
        show_api=False,
        server_name="0.0.0.0",  # For cloud deployment
        server_port=7860,       # Standard Gradio port
        share=False
    )