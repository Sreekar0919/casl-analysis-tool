import gradio as gr
import json
import os
import logging
import requests
import re
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anthropic API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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
            return_all_scores=True
        )
        
        # Emotion analysis
        emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        SPEECHBRAIN_AVAILABLE = False
        HUGGINGFACE_AVAILABLE = False

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

def transcribe_audio_with_metadata(audio_file):
    """Transcribe audio with timestamps, sentiment, and metadata"""
    if not audio_file:
        return None, "No audio file provided"
    
    if not SPEECHBRAIN_AVAILABLE:
        return None, "SpeechBrain not available - using demo transcription"
    
    try:
        # Get transcription with timestamps
        transcript = asr_model.transcribe_file(audio_file)
        
        # Split into sentences for analysis
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Analyze each sentence
        rich_transcript = []
        current_time = 0
        
        for i, sentence in enumerate(sentences):
            # Estimate timestamp (rough approximation)
            timestamp = current_time + (i * 2)  # Assume ~2 seconds per sentence
            
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
        
        return rich_transcript, "Transcription completed successfully"
        
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
        
        line = f"[{timestamp_str}] *PAR: {entry['sentence']}"
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
        'speech_rate_variability': round(np.std(speech_rates), 1) if speech_rates else 0
    }

def generate_slp_analysis_prompt(rich_transcript, metrics, age, gender, slp_notes=""):
    """Generate comprehensive SLP analysis prompt"""
    
    # Format metrics for the prompt
    metrics_text = f"""
TRANSCRIPT METRICS:
- Total sentences: {metrics['total_sentences']}
- Total words: {metrics['total_words']}
- Duration: {metrics['total_duration_seconds']:.1f} seconds
- Type-Token Ratio: {metrics['type_token_ratio']} (vocabulary diversity)
- Average sentence length: {metrics['avg_sentence_length']} words
- Average speech rate: {metrics['avg_speech_rate_wpm']} words per minute
- Speech rate variability: {metrics['speech_rate_variability']} wpm
- Average pause duration: {metrics['avg_pause_duration']:.1f} seconds

SENTIMENT DISTRIBUTION: {metrics['sentiment_distribution']}
EMOTION DISTRIBUTION: {metrics['emotion_distribution']}

MOST FREQUENT WORDS: {list(metrics['word_frequency'].keys())[:10]}
"""
    
    # Format rich transcript for analysis
    transcript_text = format_rich_transcript(rich_transcript)
    
    notes_section = f"\nSLP CLINICAL NOTES:\n{slp_notes}" if slp_notes else ""
    
    prompt = f"""
You are a speech-language pathologist conducting a comprehensive analysis of a speech transcript with rich metadata.

PATIENT: {age}-year-old {gender}

{metrics_text}

TRANSCRIPT WITH METADATA:
{transcript_text}{notes_section}

Please provide a comprehensive analysis including:

1. SPEECH FLUENCY ANALYSIS:
   - Speech rate patterns and variability
   - Pause patterns and their significance
   - Overall fluency assessment

2. LANGUAGE COMPLEXITY:
   - Vocabulary diversity and word frequency patterns
   - Sentence structure and complexity
   - Language development level assessment

3. EMOTIONAL AND AFFECTIVE ANALYSIS:
   - Sentiment patterns throughout the transcript
   - Emotional expression and regulation
   - Impact on communication effectiveness

4. SPEECH FACTORS:
   - Word retrieval patterns
   - Grammatical accuracy
   - Repetitions and revisions

5. CLINICAL IMPLICATIONS:
   - Specific intervention targets
   - Strengths and areas for improvement
   - Recommendations for therapy

6. COMPREHENSIVE SUMMARY:
   - Overall communication profile
   - Developmental appropriateness
   - Prognosis and treatment priorities

Use the quantitative metrics and qualitative observations to support your analysis.
"""
    
    return prompt

def analyze_rich_transcript(rich_transcript, age, gender, slp_notes=""):
    """Analyze rich transcript with comprehensive metrics"""
    if not rich_transcript:
        return "No transcript data available for analysis."
    
    # Calculate SLP metrics
    metrics = calculate_slp_metrics(rich_transcript)
    
    # Generate analysis prompt
    prompt = generate_slp_analysis_prompt(rich_transcript, metrics, age, gender, slp_notes)
    
    # Get analysis from Claude API
    if ANTHROPIC_API_KEY:
        result = call_claude_api(prompt)
    else:
        result = generate_demo_analysis(rich_transcript, metrics)
    
    return result

def generate_demo_analysis(rich_transcript, metrics):
    """Generate demo analysis when API is not available"""
    return f"""## Comprehensive SLP Analysis

### SPEECH FLUENCY ANALYSIS
**Speech Rate**: {metrics['avg_speech_rate_wpm']} words per minute (variability: {metrics['speech_rate_variability']} wpm)
- Speech rate appears {'within normal limits' if 120 <= metrics['avg_speech_rate_wpm'] <= 180 else 'below typical range' if metrics['avg_speech_rate_wpm'] < 120 else 'above typical range'}
- Variability suggests {'consistent' if metrics['speech_rate_variability'] < 20 else 'variable'} speech patterns

**Pause Analysis**: Average pause duration of {metrics['avg_pause_duration']:.1f} seconds
- {'Appropriate' if 0.5 <= metrics['avg_pause_duration'] <= 2.0 else 'Short' if metrics['avg_pause_duration'] < 0.5 else 'Long'} pauses between utterances

### LANGUAGE COMPLEXITY
**Vocabulary Diversity**: Type-Token Ratio of {metrics['type_token_ratio']}
- {'Good' if metrics['type_token_ratio'] > 0.4 else 'Limited' if metrics['type_token_ratio'] < 0.3 else 'Moderate'} vocabulary diversity

**Sentence Structure**: Average {metrics['avg_sentence_length']} words per sentence
- Sentence length appears {'age-appropriate' if 5 <= metrics['avg_sentence_length'] <= 12 else 'below age expectations' if metrics['avg_sentence_length'] < 5 else 'above age expectations'}

**Most Frequent Words**: {', '.join(list(metrics['word_frequency'].keys())[:5])}

### EMOTIONAL AND AFFECTIVE ANALYSIS
**Sentiment Distribution**: {metrics['sentiment_distribution']}
**Emotion Distribution**: {metrics['emotion_distribution']}

### CLINICAL IMPLICATIONS
Based on the quantitative analysis, this patient shows:
- {'Good' if metrics['type_token_ratio'] > 0.4 else 'Limited'} vocabulary diversity
- {'Appropriate' if 120 <= metrics['avg_speech_rate_wpm'] <= 180 else 'Atypical'} speech rate
- {'Consistent' if metrics['speech_rate_variability'] < 20 else 'Variable'} speech patterns

### RECOMMENDATIONS
1. Focus on vocabulary expansion if TTR < 0.4
2. Address speech rate if outside normal range
3. Work on sentence complexity if below age expectations
4. Consider emotional regulation strategies based on sentiment patterns"""

def create_enhanced_interface():
    """Create the enhanced Gradio interface"""
    with gr.Blocks(title="Enhanced CASL Analysis Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🗣️ Enhanced CASL Analysis Tool")
        gr.Markdown("Advanced speech analysis with sentiment, timestamps, and comprehensive SLP metrics")
        
        with gr.Tabs():
            # Audio Upload & Transcription Tab
            with gr.Tab("🎤 Audio Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Audio Upload")
                        
                        audio_input = gr.Audio(
                            type="filepath",
                            label="Upload Audio Recording"
                        )
                        
                        transcribe_btn = gr.Button(
                            "🎤 Transcribe & Analyze", 
                            variant="primary",
                            size="lg"
                        )
                        
                        transcription_status = gr.Markdown("")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Rich Transcript")
                        
                        rich_transcript_display = gr.Textbox(
                            label="Transcription with Timestamps & Sentiment",
                            lines=15,
                            max_lines=20
                        )
            
            # Analysis Tab
            with gr.Tab("📊 Analysis"):
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
                            "🔍 Analyze Transcript", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Comprehensive Analysis")
                        
                        analysis_output = gr.Textbox(
                            label="SLP Analysis Report",
                            lines=25,
                            max_lines=30
                        )
            
            # Metrics Tab
            with gr.Tab("📈 Metrics Dashboard"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Quantitative Metrics")
                        
                        metrics_display = gr.JSON(
                            label="SLP Metrics",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Word Frequency")
                        
                        word_freq_display = gr.Dataframe(
                            headers=["Word", "Frequency"],
                            label="Most Frequent Words",
                            interactive=False
                        )
        
        # Event handlers
        def on_transcribe(audio_file):
            """Handle audio transcription"""
            if not audio_file:
                return "", "Please upload an audio file first."
            
            rich_transcript, status = transcribe_audio_with_metadata(audio_file)
            
            if rich_transcript:
                formatted = format_rich_transcript(rich_transcript)
                return formatted, status
            else:
                return "", status
        
        def on_analyze(rich_transcript_text, age_val, gender_val, notes):
            """Handle analysis"""
            # Convert formatted text back to rich transcript structure
            # This is a simplified version - in practice you'd want to store the rich data
            if not rich_transcript_text or rich_transcript_text == "No transcript data available":
                return "Please transcribe audio first."
            
            # For demo purposes, create a simple rich transcript from the text
            lines = rich_transcript_text.split('\n')
            rich_transcript = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    # Extract sentence from the line
                    sentence_match = re.search(r'\*PAR: (.+?)(?=\s*\[|$)', line)
                    if sentence_match:
                        sentence = sentence_match.group(1).strip()
                        rich_transcript.append({
                            'timestamp': i * 2,
                            'sentence': sentence,
                            'word_count': len(sentence.split()),
                            'avg_word_length': np.mean([len(word) for word in sentence.split()]) if sentence.split() else 0,
                            'speech_rate_wpm': 120.0,
                            'sentiment': 'neutral',
                            'sentiment_score': 0.5,
                            'emotion': 'neutral',
                            'emotion_score': 0.5
                        })
            
            return analyze_rich_transcript(rich_transcript, age_val, gender_val, notes)
        
        def update_metrics(rich_transcript_text):
            """Update metrics display"""
            if not rich_transcript_text or rich_transcript_text == "No transcript data available":
                return {}, []
            
            # Convert text back to rich transcript (simplified)
            lines = rich_transcript_text.split('\n')
            rich_transcript = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    sentence_match = re.search(r'\*PAR: (.+?)(?=\s*\[|$)', line)
                    if sentence_match:
                        sentence = sentence_match.group(1).strip()
                        rich_transcript.append({
                            'timestamp': i * 2,
                            'sentence': sentence,
                            'word_count': len(sentence.split()),
                            'avg_word_length': np.mean([len(word) for word in sentence.split()]) if sentence.split() else 0,
                            'speech_rate_wpm': 120.0,
                            'sentiment': 'neutral',
                            'sentiment_score': 0.5,
                            'emotion': 'neutral',
                            'emotion_score': 0.5
                        })
            
            metrics = calculate_slp_metrics(rich_transcript)
            
            # Create word frequency dataframe
            word_freq_data = [[word, freq] for word, freq in list(metrics['word_frequency'].items())[:20]]
            
            return metrics, word_freq_data
        
        # Connect event handlers
        transcribe_btn.click(
            on_transcribe,
            inputs=[audio_input],
            outputs=[rich_transcript_display, transcription_status]
        )
        
        analyze_btn.click(
            on_analyze,
            inputs=[rich_transcript_display, age, gender, slp_notes],
            outputs=[analysis_output]
        )
        
        # Update metrics when transcript changes
        rich_transcript_display.change(
            update_metrics,
            inputs=[rich_transcript_display],
            outputs=[metrics_display, word_freq_display]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Enhanced CASL Analysis Tool...")
    
    if not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not configured - analysis will show demo response")
        print("   For HuggingFace Spaces: Add ANTHROPIC_API_KEY as a secret in your space settings")
        print("   For local use: export ANTHROPIC_API_KEY='your-key-here'")
    else:
        print("✅ Claude API configured")
    
    if not SPEECHBRAIN_AVAILABLE:
        print("⚠️  SpeechBrain not available - audio transcription will use demo mode")
        print("   Install with: pip install speechbrain transformers torch")
    else:
        print("✅ SpeechBrain and HuggingFace models loaded")
    
    app = create_enhanced_interface()
    app.launch(show_api=False) 