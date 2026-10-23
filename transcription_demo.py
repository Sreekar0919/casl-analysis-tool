import gradio as gr
import json
import os
import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime
import time
import tempfile
from typing import Dict, List, Tuple, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import video processing libraries
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
    logger.info("MoviePy available for video processing")
except ImportError as e:
    logger.warning(f"MoviePy not available: {e}")
    MOVIEPY_AVAILABLE = False

# Try to import speaker diarization
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    DIARIZATION_AVAILABLE = True
    logger.info("Pyannote.audio available for speaker diarization")
except ImportError as e:
    logger.warning(f"Pyannote.audio not available: {e}")
    DIARIZATION_AVAILABLE = False

# Try to import SpeechBrain and HuggingFace components
try:
    from speechbrain.pretrained import EncoderDecoderASR, VAD, EncoderClassifier
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    SPEECHBRAIN_AVAILABLE = True
    HUGGINGFACE_AVAILABLE = True
    logger.info("SpeechBrain and HuggingFace models available")
except ImportError as e:
    logger.warning(f"SpeechBrain/HuggingFace not available: {e}")
    SPEECHBRAIN_AVAILABLE = False
    HUGGINGFACE_AVAILABLE = False

# Initialize models if available
asr_model = None
vad_model = None
sentiment_model = None
emotion_model = None
diarization_pipeline = None

if SPEECHBRAIN_AVAILABLE and HUGGINGFACE_AVAILABLE:
    try:
        # Speech-to-text model
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech"
        )
        
        # Voice Activity Detection
        vad_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty"
        )
        
        # Sentiment analysis
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None
        )
        
        # Emotion analysis
        emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        SPEECHBRAIN_AVAILABLE = False
        HUGGINGFACE_AVAILABLE = False

# Initialize diarization pipeline
if DIARIZATION_AVAILABLE:
    try:
        # Note: You'll need to get a HuggingFace token and accept the model terms
        # at https://huggingface.co/pyannote/speaker-diarization
        HF_TOKEN = os.getenv("HF_TOKEN", "")
        if HF_TOKEN:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=HF_TOKEN
            )
            logger.info("Speaker diarization pipeline loaded")
        else:
            logger.warning("HF_TOKEN not set - speaker diarization will be disabled")
    except Exception as e:
        logger.error(f"Error loading diarization pipeline: {e}")

def extract_audio_from_video(video_path):
    """Extract audio from video file (MP4, etc.)"""
    if not MOVIEPY_AVAILABLE:
        return None, "MoviePy not available for video processing"
    
    try:
        # Create temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Load video and extract audio
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        if audio is None:
            return None, "No audio track found in video file"
        
        # Export audio to temporary WAV file
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
        # Close video to free memory
        video.close()
        audio.close()
        
        logger.info(f"Audio extracted from video: {temp_audio_path}")
        return temp_audio_path, "Audio extracted successfully"
        
    except Exception as e:
        logger.error(f"Error extracting audio from video: {e}")
        return None, f"Error extracting audio: {str(e)}"

def perform_speaker_diarization(audio_path):
    """Perform speaker diarization on audio file"""
    if not DIARIZATION_AVAILABLE or not diarization_pipeline:
        return None, "Speaker diarization not available"
    
    try:
        # Perform diarization
        with ProgressHook() as hook:
            diarization = diarization_pipeline(audio_path, hook=hook)
        
        # Extract speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })
        
        logger.info(f"Diarization completed: {len(speaker_segments)} segments found")
        return speaker_segments, "Diarization completed successfully"
        
    except Exception as e:
        logger.error(f"Error in diarization: {e}")
        return None, f"Diarization error: {str(e)}"

def process_audio_file(file_path):
    """Process audio file, extracting from video if needed"""
    if not file_path:
        return None, "No file provided"
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # If it's a video file, extract audio first
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
        logger.info(f"Processing video file: {file_path}")
        audio_path, status = extract_audio_from_video(file_path)
        if audio_path:
            return audio_path, f"Video processed: {status}"
        else:
            return None, status
    
    # If it's already an audio file, use it directly
    elif file_extension in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        logger.info(f"Processing audio file: {file_path}")
        return file_path, "Audio file ready for transcription"
    
    else:
        return None, f"Unsupported file format: {file_extension}"

def transcribe_audio_with_metadata(audio_file, enable_diarization=True):
    """Transcribe audio with timestamps, sentiment, and metadata"""
    if not audio_file:
        return None, "No audio file provided"
    
    if not SPEECHBRAIN_AVAILABLE:
        return None, "SpeechBrain not available - using demo transcription"
    
    try:
        # Process the file (extract audio if it's a video)
        processed_audio_path, process_status = process_audio_file(audio_file)
        
        if not processed_audio_path:
            return None, process_status
        
        # Perform speaker diarization if enabled
        speaker_segments = None
        diarization_status = ""
        if enable_diarization:
            speaker_segments, diarization_status = perform_speaker_diarization(processed_audio_path)
        
        # Get transcription with timestamps
        transcript = asr_model.transcribe_file(processed_audio_path)
        
        # Clean up temporary audio file if it was created from video
        if processed_audio_path != audio_file and os.path.exists(processed_audio_path):
            try:
                os.unlink(processed_audio_path)
                logger.info("Temporary audio file cleaned up")
            except Exception as e:
                logger.warning(f"Could not clean up temporary file: {e}")
        
        # Split into sentences for analysis
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Analyze each sentence
        rich_transcript = []
        current_time = 0
        
        for i, sentence in enumerate(sentences):
            # Estimate timestamp (rough approximation)
            timestamp = current_time + (i * 2)  # Assume ~2 seconds per sentence
            
            # Determine speaker for this timestamp
            speaker = "UNKNOWN"
            if speaker_segments:
                for segment in speaker_segments:
                    if segment['start'] <= timestamp <= segment['end']:
                        speaker = segment['speaker']
                        break
            
            # Sentiment analysis
            sentiment_result = sentiment_model(sentence)[0] if sentiment_model else None
            sentiment = max(sentiment_result, key=lambda x: x['score']) if sentiment_result else {'label': 'neutral', 'score': 0.5}
            
            # Emotion analysis
            emotion_result = emotion_model(sentence)[0] if emotion_model else None
            emotion = max(emotion_result, key=lambda x: x['score']) if emotion_result else {'label': 'neutral', 'score': 0.5}
            
            # Word count and complexity metrics
            words = sentence.split()
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Calculate speech rate (words per minute estimate)
            speech_rate = word_count * 30 / 60  # Rough estimate
            
            rich_transcript.append({
                'timestamp': timestamp,
                'speaker': speaker,
                'sentence': sentence,
                'word_count': word_count,
                'avg_word_length': round(avg_word_length, 2),
                'speech_rate_wpm': round(speech_rate, 1),
                'sentiment': sentiment['label'],
                'sentiment_score': round(sentiment['score'], 3),
                'emotion': emotion['label'],
                'emotion_score': round(emotion['score'], 3)
            })
            
            current_time = timestamp
        
        status_msg = f"Transcription completed successfully. {process_status}"
        if diarization_status:
            status_msg += f" {diarization_status}"
        
        return rich_transcript, status_msg
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return None, f"Transcription error: {str(e)}"

def format_rich_transcript(rich_transcript):
    """Format rich transcript for display"""
    if not rich_transcript:
        return "No transcript data available"
    
    formatted_lines = []
    for entry in rich_transcript:
        timestamp_str = f"{int(entry['timestamp']//60):02d}:{int(entry['timestamp']%60):02d}"
        
        line = f"[{timestamp_str}] *{entry['speaker']}: {entry['sentence']}"
        line += f" [Words: {entry['word_count']}, Rate: {entry['speech_rate_wpm']}wpm]"
        line += f" [Sentiment: {entry['sentiment']} ({entry['sentiment_score']})]"
        line += f" [Emotion: {entry['emotion']} ({entry['emotion_score']})]"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def calculate_slp_metrics(rich_transcript):
    """Calculate comprehensive SLP metrics"""
    if not rich_transcript:
        return {}
    
    # Basic metrics
    total_sentences = len(rich_transcript)
    total_words = sum(entry['word_count'] for entry in rich_transcript)
    total_duration = rich_transcript[-1]['timestamp'] if rich_transcript else 0
    
    # Speaker analysis
    speakers = {}
    for entry in rich_transcript:
        speaker = entry['speaker']
        if speaker not in speakers:
            speakers[speaker] = {
                'sentences': 0,
                'words': 0,
                'sentiments': [],
                'emotions': []
            }
        speakers[speaker]['sentences'] += 1
        speakers[speaker]['words'] += entry['word_count']
        speakers[speaker]['sentiments'].append(entry['sentiment'])
        speakers[speaker]['emotions'].append(entry['emotion'])
    
    # Word-level analysis
    all_words = []
    for entry in rich_transcript:
        words = entry['sentence'].lower().split()
        all_words.extend(words)
    
    # Word frequency distribution
    word_freq = {}
    for word in all_words:
        word_clean = re.sub(r'[^\w\s]', '', word)
        if word_clean:
            word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
    
    # Vocabulary diversity (Type-Token Ratio)
    unique_words = len(set(all_words))
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Speech rate analysis
    speech_rates = [entry['speech_rate_wpm'] for entry in rich_transcript]
    avg_speech_rate = np.mean(speech_rates) if speech_rates else 0
    
    # Sentiment analysis
    sentiment_counts = {}
    emotion_counts = {}
    for entry in rich_transcript:
        sentiment_counts[entry['sentiment']] = sentiment_counts.get(entry['sentiment'], 0) + 1
        emotion_counts[entry['emotion']] = emotion_counts.get(entry['emotion'], 0) + 1
    
    # Sentence complexity
    sentence_lengths = [entry['word_count'] for entry in rich_transcript]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    
    # Pause analysis (gaps between sentences)
    pauses = []
    for i in range(1, len(rich_transcript)):
        pause = rich_transcript[i]['timestamp'] - rich_transcript[i-1]['timestamp']
        pauses.append(pause)
    
    avg_pause_duration = np.mean(pauses) if pauses else 0
    
    return {
        'total_sentences': total_sentences,
        'total_words': total_words,
        'total_duration_seconds': total_duration,
        'unique_words': unique_words,
        'type_token_ratio': round(ttr, 3),
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_speech_rate_wpm': round(avg_speech_rate, 1),
        'avg_pause_duration': round(avg_pause_duration, 1),
        'sentiment_distribution': sentiment_counts,
        'emotion_distribution': emotion_counts,
        'word_frequency': dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]),
        'speech_rate_variability': round(np.std(speech_rates), 1) if speech_rates else 0,
        'speakers': speakers,
        'speaker_count': len(speakers)
    }

def generate_comprehensive_analysis_prompt(rich_transcript, metrics, age, gender, slp_notes=""):
    """Generate comprehensive analysis prompt using rich transcript data"""
    
    # Format rich transcript with timestamps and metadata
    transcript_lines = []
    for entry in rich_transcript:
        timestamp_str = f"{int(entry['timestamp']//60):02d}:{int(entry['timestamp']%60):02d}"
        transcript_lines.append(f"[{timestamp_str}] *{entry['speaker']}: {entry['sentence']}")
    
    transcript_text = '\n'.join(transcript_lines)
    
    # Format metrics for analysis
    metrics_text = f"""
TRANSCRIPT METRICS:
• Total sentences: {metrics['total_sentences']}
• Total words: {metrics['total_words']}
• Duration: {metrics['total_duration_seconds']:.1f} seconds
• Type-Token Ratio: {metrics['type_token_ratio']} (vocabulary diversity)
• Average sentence length: {metrics['avg_sentence_length']} words
• Average speech rate: {metrics['avg_speech_rate_wpm']} words per minute
• Speech rate variability: {metrics['speech_rate_variability']} wpm
• Average pause duration: {metrics['avg_pause_duration']:.1f} seconds
• Number of speakers: {metrics['speaker_count']}

SENTIMENT DISTRIBUTION: {metrics['sentiment_distribution']}
EMOTION DISTRIBUTION: {metrics['emotion_distribution']}

SPEAKER ANALYSIS:"""
    
    for speaker, data in metrics['speakers'].items():
        metrics_text += f"\n• {speaker}: {data['sentences']} sentences, {data['words']} words"
    
    metrics_text += f"\n\nMOST FREQUENT WORDS: {list(metrics['word_frequency'].keys())[:10]}"
    
    notes_section = f"\nSLP CLINICAL NOTES:\n{slp_notes}" if slp_notes else ""
    
    prompt = f"""
You are a speech-language pathologist conducting a comprehensive analysis of a speech transcript with rich temporal and affective metadata.

PATIENT: {age}-year-old {gender}

{metrics_text}

TRANSCRIPT WITH TIMESTAMPS AND METADATA:
{transcript_text}{notes_section}

Please provide a comprehensive analysis including:

1. TEMPORAL SPEECH PATTERNS:
   - Analyze speech rate changes over time using timestamps
   - Identify patterns in pause duration and frequency
   - Assess temporal consistency in speech production
   - Note any significant changes in speech patterns throughout the session

2. AFFECTIVE AND EMOTIONAL ANALYSIS:
   - Analyze sentiment patterns throughout the transcript using timestamp data
   - Identify emotional shifts and their potential causes
   - Assess emotional regulation and expression
   - Note any correlations between emotional state and speech characteristics

3. SPEAKER-SPECIFIC ANALYSIS (if multiple speakers):
   - Compare speech patterns between speakers
   - Analyze turn-taking patterns and timing
   - Assess interaction dynamics
   - Note speaker-specific emotional and sentiment patterns

4. SPEECH FLUENCY AND RATE ANALYSIS:
   - Analyze speech rate variability using the provided metrics
   - Identify periods of fluent vs. dysfluent speech
   - Assess the impact of emotional state on speech rate
   - Note any temporal patterns in speech rate changes

5. LANGUAGE COMPLEXITY ASSESSMENT:
   - Analyze vocabulary diversity using Type-Token Ratio
   - Assess sentence complexity and variety
   - Identify patterns in word frequency and usage
   - Note any temporal changes in language complexity

6. COMPLEX SENTENCE ANALYSIS:
   - Count and analyze use of coordinating conjunctions (and, but, or, so, yet, for, nor)
   - Count and analyze use of subordinating conjunctions (because, although, while, since, if, when, where, that, which, who, whom, whose)
   - Identify compound, complex, and compound-complex sentences
   - Assess sentence variety and complexity level for age

7. FIGURATIVE LANGUAGE ANALYSIS:
   - Identify and count similes (comparisons using "like" or "as")
   - Identify and count metaphors (direct comparisons without "like" or "as")
   - Identify and count idioms (common expressions with non-literal meanings)
   - Assess figurative language comprehension and use for age

8. CLINICAL IMPLICATIONS:
   - Specific intervention targets based on temporal patterns
   - Recommendations for emotional regulation if needed
   - Suggestions for improving speech rate consistency
   - Strategies for enhancing language complexity
   - Age-appropriate development recommendations

9. COMPREHENSIVE SUMMARY:
   - Overall communication profile with temporal considerations
   - Assessment of emotional and affective communication
   - Developmental appropriateness considering age
   - Prognosis and treatment priorities

Use the temporal data, sentiment scores, and emotional labels to provide insights that would not be possible with a simple transcript. Reference specific timestamps and emotional states when making observations.
"""
    
    return prompt

def analyze_rich_transcript_with_llm(rich_transcript, age, gender, slp_notes=""):
    """Analyze rich transcript using LLM with comprehensive metadata"""
    if not rich_transcript:
        return "No transcript data available for analysis."
    
    # Calculate SLP metrics
    metrics = calculate_slp_metrics(rich_transcript)
    
    # Generate comprehensive analysis prompt
    prompt = generate_comprehensive_analysis_prompt(rich_transcript, metrics, age, gender, slp_notes)
    
    # Get analysis from Claude API
    if ANTHROPIC_API_KEY:
        result = call_claude_api(prompt)
    else:
        result = generate_demo_analysis(rich_transcript, metrics)
    
    return result

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

def generate_demo_analysis(rich_transcript, metrics):
    """Generate demo analysis when API is not available"""
    return f"""## Comprehensive SLP Analysis with Temporal and Affective Data

### TEMPORAL SPEECH PATTERNS
**Speech Rate Analysis**: {metrics['avg_speech_rate_wpm']} words per minute (variability: {metrics['speech_rate_variability']} wpm)
- Speech rate appears {'within normal limits' if 120 <= metrics['avg_speech_rate_wpm'] <= 180 else 'below typical range' if metrics['avg_speech_rate_wpm'] < 120 else 'above typical range'}
- Variability suggests {'consistent' if metrics['speech_rate_variability'] < 20 else 'variable'} speech patterns

**Pause Analysis**: Average pause duration of {metrics['avg_pause_duration']:.1f} seconds
- {'Appropriate' if 0.5 <= metrics['avg_pause_duration'] <= 2.0 else 'Short' if metrics['avg_pause_duration'] < 0.5 else 'Long'} pauses between utterances

### AFFECTIVE AND EMOTIONAL ANALYSIS
**Sentiment Distribution**: {metrics['sentiment_distribution']}
**Emotion Distribution**: {metrics['emotion_distribution']}

The emotional patterns suggest {'positive' if 'positive' in metrics['sentiment_distribution'] and metrics['sentiment_distribution']['positive'] > 2 else 'neutral' if 'neutral' in metrics['sentiment_distribution'] and metrics['sentiment_distribution']['neutral'] > 2 else 'mixed'} emotional expression throughout the session.

### LANGUAGE COMPLEXITY
**Vocabulary Diversity**: Type-Token Ratio of {metrics['type_token_ratio']}
- {'Good' if metrics['type_token_ratio'] > 0.4 else 'Limited' if metrics['type_token_ratio'] < 0.3 else 'Moderate'} vocabulary diversity

**Sentence Structure**: Average {metrics['avg_sentence_length']} words per sentence
- Sentence length appears {'age-appropriate' if 5 <= metrics['avg_sentence_length'] <= 12 else 'below age expectations' if metrics['avg_sentence_length'] < 5 else 'above age expectations'}

**Most Frequent Words**: {', '.join(list(metrics['word_frequency'].keys())[:5])}

### SPEAKER ANALYSIS
**Number of Speakers**: {metrics['speaker_count']}
{chr(10).join([f"• {speaker}: {data['sentences']} sentences, {data['words']} words" for speaker, data in metrics['speakers'].items()])}

### CLINICAL IMPLICATIONS
Based on the temporal and affective analysis, this patient shows:
- {'Good' if metrics['type_token_ratio'] > 0.4 else 'Limited'} vocabulary diversity
- {'Appropriate' if 120 <= metrics['avg_speech_rate_wpm'] <= 180 else 'Atypical'} speech rate
- {'Consistent' if metrics['speech_rate_variability'] < 20 else 'Variable'} speech patterns
- {'Positive' if 'positive' in metrics['sentiment_distribution'] and metrics['sentiment_distribution']['positive'] > 2 else 'Neutral'} emotional expression

### RECOMMENDATIONS
1. Focus on vocabulary expansion if TTR < 0.4
2. Address speech rate if outside normal range
3. Work on sentence complexity if below age expectations
4. Consider emotional regulation strategies based on sentiment patterns
5. Monitor temporal patterns in speech rate and fluency"""

def create_transcription_interface():
    """Create the transcription-focused Gradio interface"""
    with gr.Blocks(title="Advanced Transcription Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎤 Advanced Transcription Tool")
        gr.Markdown("Transcribe audio/video with speaker diarization, timestamps, sentiment analysis, and comprehensive LLM analysis")
        
        with gr.Tabs():
            # Audio/Video Upload & Transcription Tab
            with gr.Tab("🎤 Audio/Video Transcription"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### File Upload")
                        gr.Markdown("**Supported formats:** MP4, AVI, MOV, MKV, WMV, FLV, WAV, MP3, M4A, FLAC, OGG")
                        
                        file_input = gr.File(
                            label="Upload Audio or Video File",
                            file_types=["audio", "video"]
                        )
                        
                        enable_diarization = gr.Checkbox(
                            label="Enable Speaker Diarization",
                            value=True,
                            info="Identify different speakers in the audio"
                        )
                        
                        transcribe_btn = gr.Button(
                            "🎤 Transcribe File", 
                            variant="primary",
                            size="lg"
                        )
                        
                        transcription_status = gr.Markdown("")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Rich Transcript with Metadata")
                        
                        rich_transcript_display = gr.Textbox(
                            label="Transcription with Speakers, Timestamps, Sentiment & Emotion",
                            lines=15,
                            max_lines=20
                        )
            
            # Analysis Tab
            with gr.Tab("📊 LLM Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Patient Information")
                        
                        with gr.Row():
                            age = gr.Number(label="Age", value=8, minimum=1, maximum=120)
                            gender = gr.Radio(["male", "female", "other"], label="Gender", value="male")
                        
                        slp_notes = gr.Textbox(
                            label="SLP Clinical Notes (Optional)",
                            placeholder="Enter additional clinical observations...",
                            lines=3
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze with LLM", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Comprehensive LLM Analysis")
                        
                        analysis_output = gr.Textbox(
                            label="LLM Analysis Report",
                            lines=25,
                            max_lines=30
                        )
            
            # Metrics Tab
            with gr.Tab("📈 Speech Metrics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Quantitative Speech Metrics")
                        
                        metrics_display = gr.Textbox(
                            label="SLP Metrics",
                            lines=15,
                            max_lines=20
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Word Frequency Analysis")
                        
                        word_freq_display = gr.Dataframe(
                            headers=["Word", "Frequency"],
                            label="Most Frequent Words",
                            interactive=False
                        )
            
            # Raw Data Tab
            with gr.Tab("📊 Raw Data"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### JSON Data")
                        
                        json_display = gr.Textbox(
                            label="Raw JSON Data",
                            lines=20,
                            max_lines=25
                        )
        
        # Event handlers
        def on_transcribe(file, diarization_enabled):
            """Handle file transcription"""
            if not file:
                return "", "", "", "Please upload a file first."
            
            rich_transcript, status = transcribe_audio_with_metadata(file.name, diarization_enabled)
            
            if rich_transcript:
                formatted = format_rich_transcript(rich_transcript)
                metrics = calculate_slp_metrics(rich_transcript)
                
                # Format metrics for display
                metrics_text = f"""SPEECH METRICS:
• Total sentences: {metrics['total_sentences']}
• Total words: {metrics['total_words']}
• Duration: {metrics['total_duration_seconds']:.1f} seconds
• Type-Token Ratio: {metrics['type_token_ratio']} (vocabulary diversity)
• Average sentence length: {metrics['avg_sentence_length']} words
• Average speech rate: {metrics['avg_speech_rate_wpm']} words per minute
• Speech rate variability: {metrics['speech_rate_variability']} wpm
• Average pause duration: {metrics['avg_pause_duration']:.1f} seconds
• Number of speakers: {metrics['speaker_count']}

SENTIMENT DISTRIBUTION: {metrics['sentiment_distribution']}
EMOTION DISTRIBUTION: {metrics['emotion_distribution']}

SPEAKER ANALYSIS:"""
                
                for speaker, data in metrics['speakers'].items():
                    metrics_text += f"\n• {speaker}: {data['sentences']} sentences, {data['words']} words"
                
                # Create word frequency dataframe
                word_freq_data = [[word, freq] for word, freq in list(metrics['word_frequency'].items())[:20]]
                
                # JSON data
                json_data = json.dumps(rich_transcript, indent=2)
                
                return formatted, metrics_text, word_freq_data, status
            else:
                return "", "", [], status
        
        def on_analyze(rich_transcript_text, age_val, gender_val, notes):
            """Handle LLM analysis"""
            if not rich_transcript_text or rich_transcript_text == "No transcript data available":
                return "Please transcribe audio first."
            
            # Convert formatted text back to rich transcript structure
            lines = rich_transcript_text.split('\n')
            rich_transcript = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    # Extract data from the formatted line
                    timestamp_match = re.search(r'\[(\d{2}:\d{2})\]', line)
                    speaker_match = re.search(r'\*(\w+):', line)
                    sentence_match = re.search(r'\*\w+:\s*(.+?)(?=\s*\[|$)', line)
                    
                    if timestamp_match and speaker_match and sentence_match:
                        timestamp_str = timestamp_match.group(1)
                        minutes, seconds = map(int, timestamp_str.split(':'))
                        timestamp = minutes * 60 + seconds
                        
                        speaker = speaker_match.group(1)
                        sentence = sentence_match.group(1).strip()
                        
                        rich_transcript.append({
                            'timestamp': timestamp,
                            'speaker': speaker,
                            'sentence': sentence,
                            'word_count': len(sentence.split()),
                            'avg_word_length': np.mean([len(word) for word in sentence.split()]) if sentence.split() else 0,
                            'speech_rate_wpm': 120.0,
                            'sentiment': 'neutral',
                            'sentiment_score': 0.5,
                            'emotion': 'neutral',
                            'emotion_score': 0.5
                        })
            
            return analyze_rich_transcript_with_llm(rich_transcript, age_val, gender_val, notes)
        
        # Connect event handlers
        transcribe_btn.click(
            on_transcribe,
            inputs=[file_input, enable_diarization],
            outputs=[rich_transcript_display, metrics_display, word_freq_display, transcription_status]
        )
        
        analyze_btn.click(
            on_analyze,
            inputs=[rich_transcript_display, age, gender, slp_notes],
            outputs=[analysis_output]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Advanced Transcription Tool...")
    
    if not MOVIEPY_AVAILABLE:
        print("⚠️  MoviePy not available - video processing will be limited")
        print("   Install with: pip install moviepy")
    else:
        print("✅ MoviePy available for video processing")
    
    if not DIARIZATION_AVAILABLE:
        print("⚠️  Pyannote.audio not available - speaker diarization will be disabled")
        print("   Install with: pip install pyannote.audio")
    else:
        print("✅ Pyannote.audio available for speaker diarization")
        if not os.getenv("HF_TOKEN"):
            print("⚠️  HF_TOKEN not set - set it to enable speaker diarization")
            print("   Get token from: https://huggingface.co/settings/tokens")
            print("   Accept model terms at: https://huggingface.co/pyannote/speaker-diarization")
    
    if not SPEECHBRAIN_AVAILABLE:
        print("⚠️  SpeechBrain not available - audio transcription will use demo mode")
        print("   Install with: pip install speechbrain transformers torch")
    else:
        print("✅ SpeechBrain and HuggingFace models loaded")
    
    app = create_transcription_interface()
    app.launch(show_api=False) 