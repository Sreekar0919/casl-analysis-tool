import gradio as gr
import json
import os
import logging
import requests
import re
import tempfile
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anthropic API key - can be set as HuggingFace secret or environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Try to import transcription libraries
try:
    from speechbrain.pretrained import EncoderDecoderASR
    import torch
    SPEECHBRAIN_AVAILABLE = True
    logger.info("SpeechBrain available for transcription")
except ImportError as e:
    logger.warning(f"SpeechBrain not available: {e}")
    SPEECHBRAIN_AVAILABLE = False

# Try to import video processing
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

# Try to import sentiment and emotion analysis
try:
    from transformers import pipeline
    SENTIMENT_AVAILABLE = True
    logger.info("Transformers available for sentiment analysis")
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    SENTIMENT_AVAILABLE = False

# Initialize models if available
asr_model = None
sentiment_model = None
emotion_model = None
diarization_pipeline = None

if SPEECHBRAIN_AVAILABLE:
    try:
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech"
        )
        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ASR model: {e}")
        SPEECHBRAIN_AVAILABLE = False

if SENTIMENT_AVAILABLE:
    try:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None
        )
        emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        logger.info("Sentiment and emotion models loaded")
    except Exception as e:
        logger.error(f"Error loading sentiment models: {e}")
        SENTIMENT_AVAILABLE = False

if DIARIZATION_AVAILABLE:
    try:
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

# Check if API key is available
if ANTHROPIC_API_KEY:
    logger.info("Claude API key found")
else:
    logger.warning("Claude API key not found - using demo mode")

def segment_response_by_sections(response_text):
    """Segment response by section titles and return a dictionary of sections"""
    required_sections = [
        "1. SPEECH FACTORS",
        "2. LANGUAGE SKILLS ASSESSMENT", 
        "3. COMPLEX SENTENCE ANALYSIS",
        "4. FIGURATIVE LANGUAGE ANALYSIS",
        "5. PRAGMATIC LANGUAGE ASSESSMENT",
        "6. VOCABULARY AND SEMANTIC ANALYSIS",
        "7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS",
        "8. COGNITIVE-LINGUISTIC FACTORS",
        "9. FLUENCY AND RHYTHM ANALYSIS",
        "10. QUANTITATIVE METRICS",
        "11. CLINICAL IMPLICATIONS",
        "12. PROGNOSIS AND SUMMARY"
    ]
    
    sections = {}
    current_section = None
    current_content = []
    
    lines = response_text.split('\n')
    
    for line in lines:
        # Check if this line is a section header
        is_section_header = False
        for section in required_sections:
            if section in line:
                # Save previous section if exists
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = section
                current_content = []
                is_section_header = True
                break
        
        # If not a section header, add to current section content
        if not is_section_header and current_section:
            current_content.append(line)
    
    # Save the last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

def combine_sections_smartly(sections_dict):
    """Combine sections in the correct order without duplicates"""
    required_sections = [
        "1. SPEECH FACTORS",
        "2. LANGUAGE SKILLS ASSESSMENT", 
        "3. COMPLEX SENTENCE ANALYSIS",
        "4. FIGURATIVE LANGUAGE ANALYSIS",
        "5. PRAGMATIC LANGUAGE ASSESSMENT",
        "6. VOCABULARY AND SEMANTIC ANALYSIS",
        "7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS",
        "8. COGNITIVE-LINGUISTIC FACTORS",
        "9. FLUENCY AND RHYTHM ANALYSIS",
        "10. QUANTITATIVE METRICS",
        "11. CLINICAL IMPLICATIONS",
        "12. PROGNOSIS AND SUMMARY"
    ]
    
    combined_parts = []
    combined_parts.append("COMPREHENSIVE SPEECH SAMPLE ANALYSIS")
    combined_parts.append("")
    
    for section in required_sections:
        if section in sections_dict:
            combined_parts.append(section)
            combined_parts.append("")
            combined_parts.append(sections_dict[section])
            combined_parts.append("")
    
    return '\n'.join(combined_parts)

def call_claude_api_with_continuation(prompt, max_continuations=0):
    """Call Claude API with smart continuation system - unlimited continuations until complete"""
    if not ANTHROPIC_API_KEY:
        return "❌ Claude API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    # Define all required sections
    required_sections = [
        "1. SPEECH FACTORS",
        "2. LANGUAGE SKILLS ASSESSMENT", 
        "3. COMPLEX SENTENCE ANALYSIS",
        "4. FIGURATIVE LANGUAGE ANALYSIS",
        "5. PRAGMATIC LANGUAGE ASSESSMENT",
        "6. VOCABULARY AND SEMANTIC ANALYSIS",
        "7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS",
        "8. COGNITIVE-LINGUISTIC FACTORS",
        "9. FLUENCY AND RHYTHM ANALYSIS",
        "10. QUANTITATIVE METRICS",
        "11. CLINICAL IMPLICATIONS",
        "12. PROGNOSIS AND SUMMARY"
    ]
    
    # Safety limits to prevent infinite loops
    MAX_CONTINUATIONS = 30  # Increased from 20 to 30 API calls
    MAX_TIME_MINUTES = 15   # Increased from 10 to 15 minutes total
    MIN_PROGRESS_PER_CALL = 0  # Changed from 1 to 0 to allow more flexibility
    
    try:
        all_sections = {}  # Store all sections found across all parts
        continuation_count = 0
        start_time = time.time()
        last_section_count = 0  # Track progress between calls
        
        # Add continuation instruction to original prompt
        initial_prompt = prompt + "\n\nCRITICAL INSTRUCTIONS: You MUST complete ALL 12 sections of the analysis. If your response is cut off or incomplete, end with <CONTINUE> to indicate more content is needed. Do not skip any sections. Use the checklist to ensure all sections are completed."
        
        while True:  # Unlimited continuations until complete
            if continuation_count == 0:
                current_prompt = initial_prompt
            else:
                # For continuations, provide context about what was already covered
                missing_sections = [s for s in required_sections if s not in all_sections]
                missing_text = "\n".join([f"- {section}" for section in missing_sections])
                
                current_prompt = prompt + f"\n\nCONTINUATION {continuation_count + 1}: The following sections are STILL MISSING and MUST be completed:\n\n{missing_text}\n\nCRITICAL: Provide ONLY these missing sections. Do not repeat any sections that are already complete. Focus exclusively on the missing sections listed above. Complete ALL missing sections in this response."
            
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
                        "content": current_prompt
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=90
            )
            
            if response.status_code == 200:
                response_json = response.json()
                response_text = response_json['content'][0]['text']
                
                # Log response for debugging
                print(f"\n=== PART {continuation_count + 1} RESPONSE ===")
                print(f"Length: {len(response_text)} characters")
                print(f"Contains CONTINUE: {'<CONTINUE>' in response_text}")
                print(f"First 200 chars: {response_text[:200]}...")
                print(f"Last 200 chars: {response_text[-200:]}...")
                print("=" * 50)
                
                # Segment this part and add new sections to our collection
                part_sections = segment_response_by_sections(response_text)
                for section, content in part_sections.items():
                    if section not in all_sections:  # Only add if not already present
                        all_sections[section] = content
                        print(f"Added section: {section}")
                    else:
                        print(f"Skipped duplicate section: {section}")
                
                # Check completion status
                completed_sections = len(all_sections)
                missing_sections = [s for s in required_sections if s not in all_sections]
                
                print(f"Completed sections: {completed_sections}/12")
                print(f"Missing sections: {missing_sections}")
                
                # Check if response indicates continuation is needed
                needs_continuation = "<CONTINUE>" in response_text
                
                print(f"Needs continuation: {needs_continuation}")
                print(f"Continuation count: {continuation_count}")
                
                # Safety checks to prevent infinite loops
                current_time = time.time()
                elapsed_minutes = (current_time - start_time) / 60
                current_section_count = len(all_sections)
                progress_made = current_section_count - last_section_count
                
                # Check if we're making progress
                if continuation_count > 0 and progress_made < MIN_PROGRESS_PER_CALL:
                    # Only stop if we've made multiple calls with no progress
                    if continuation_count > 3:  # Allow more attempts before giving up
                        logger.warning(f"No progress made in last call (added {progress_made} sections). Stopping to prevent infinite loop.")
                        break
                    else:
                        logger.info(f"No progress in call {continuation_count}, but continuing to allow more attempts...")
                
                # Check time limit
                if elapsed_minutes > MAX_TIME_MINUTES:
                    logger.warning(f"Time limit exceeded ({elapsed_minutes:.1f} minutes). Stopping to prevent excessive API usage.")
                    break
                
                # Check continuation limit
                if continuation_count >= MAX_CONTINUATIONS:
                    logger.warning(f"Continuation limit reached ({MAX_CONTINUATIONS} calls). Stopping to prevent excessive API usage.")
                    break
                
                # Continue if <CONTINUE> is present and safety checks pass
                if needs_continuation:
                    continuation_count += 1
                    last_section_count = current_section_count
                    logger.info(f"Continuing analysis (attempt {continuation_count}/{MAX_CONTINUATIONS}, {elapsed_minutes:.1f} minutes elapsed)")
                    continue
                else:
                    break
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return f"❌ Claude API Error: {response.status_code}"
                
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return f"❌ Error: {str(e)}"
    
    # Combine all sections in the correct order
    final_response = combine_sections_smartly(all_sections)
    
    # Log final results
    print(f"\n=== FINAL SMART VALIDATION ===")
    print(f"Total sections found: {len(all_sections)}")
    print(f"All sections present: {len(all_sections) == 12}")
    print(f"Missing sections: {[s for s in required_sections if s not in all_sections]}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Total API calls: {continuation_count + 1}")
    print("=" * 50)
    
    # Add completion indicator with safety info
    if continuation_count > 0:
        final_response += f"\n\n[Analysis completed in {continuation_count + 1} parts over {(time.time() - start_time) / 60:.1f} minutes]"
    
    # Add warning if incomplete due to safety limits
    if len(all_sections) < 12:
        missing_sections = [s for s in required_sections if s not in all_sections]
        final_response += f"\n\n⚠️ WARNING: Analysis incomplete due to safety limits. Missing sections: {', '.join(missing_sections)}"
        final_response += f"\n\n💡 TIP: Try running the analysis again, or use the 'Targeted Analysis' tab to focus on specific areas."
        final_response += f"\nThe 'Quick Questions' tab may also provide faster results for specific areas of interest."
    
    return final_response

def call_claude_api_quick_analysis(prompt):
    """Call Claude API for quick focused analysis - single response only"""
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
            timeout=90
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

def call_claude_api(prompt):
    """Call Claude API directly (legacy function for backward compatibility)"""
    return call_claude_api_quick_analysis(prompt)

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    if not MOVIEPY_AVAILABLE:
        return None, "MoviePy not available for video processing"
    
    try:
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        if audio is None:
            return None, "No audio track found in video file"
        
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()
        audio.close()
        
        return temp_audio_path, "Audio extracted successfully"
        
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None, f"Error extracting audio: {str(e)}"

def perform_speaker_diarization(audio_path):
    """Perform speaker diarization on audio file"""
    if not DIARIZATION_AVAILABLE or not diarization_pipeline:
        return None, "Speaker diarization not available"
    
    try:
        with ProgressHook() as hook:
            diarization = diarization_pipeline(audio_path, hook=hook)
        
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

def transcribe_audio(audio_file):
    """Simple transcription function for basic audio/video files"""
    if not audio_file:
        return None, "No audio file provided"
    
    if not SPEECHBRAIN_AVAILABLE:
        return None, "SpeechBrain not available for transcription"
    
    try:
        # Check if it's a video file
        file_extension = os.path.splitext(audio_file)[1].lower()
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            processed_audio, status = extract_audio_from_video(audio_file)
            if not processed_audio:
                return None, status
        else:
            processed_audio = audio_file
        
        # Get transcription
        transcript = asr_model.transcribe_file(processed_audio)
        
        # Clean up temporary file if created
        if processed_audio != audio_file and os.path.exists(processed_audio):
            try:
                os.unlink(processed_audio)
            except:
                pass
        
        return transcript, "Transcription completed successfully"
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return None, f"Transcription error: {str(e)}"

def transcribe_audio_with_metadata(audio_file, enable_diarization=True):
    """Transcribe audio with timestamps, sentiment, and metadata"""
    if not audio_file:
        return None, "No audio file provided"
    
    if not SPEECHBRAIN_AVAILABLE:
        return None, "SpeechBrain not available for transcription"
    
    try:
        # Check if it's a video file
        file_extension = os.path.splitext(audio_file)[1].lower()
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            processed_audio, status = extract_audio_from_video(audio_file)
            if not processed_audio:
                return None, status
        else:
            processed_audio = audio_file
        
        # Perform speaker diarization if enabled
        speaker_segments = None
        diarization_status = ""
        if enable_diarization:
            speaker_segments, diarization_status = perform_speaker_diarization(processed_audio)
        
        # Get transcription
        transcript = asr_model.transcribe_file(processed_audio)
        
        # Clean up temporary file if created
        if processed_audio != audio_file and os.path.exists(processed_audio):
            try:
                os.unlink(processed_audio)
            except:
                pass
        
        # Split into sentences and add metadata
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        rich_transcript = []
        current_time = 0
        
        for i, sentence in enumerate(sentences):
            timestamp = current_time + (i * 2)
            
            # Determine speaker
            speaker = "UNKNOWN"
            if speaker_segments:
                for segment in speaker_segments:
                    if segment['start'] <= timestamp <= segment['end']:
                        speaker = segment['speaker']
                        break
            
            # Sentiment and emotion analysis
            sentiment = {'label': 'neutral', 'score': 0.5}
            emotion = {'label': 'neutral', 'score': 0.5}
            
            if SENTIMENT_AVAILABLE:
                try:
                    sentiment_result = sentiment_model(sentence)[0] if sentiment_model else None
                    sentiment = max(sentiment_result, key=lambda x: x['score']) if sentiment_result else sentiment
                    
                    emotion_result = emotion_model(sentence)[0] if emotion_model else None
                    emotion = max(emotion_result, key=lambda x: x['score']) if emotion_result else emotion
                except:
                    pass
            
            # Word metrics
            words = sentence.split()
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            speech_rate = word_count * 30 / 60
            
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
        
        status_msg = f"Transcription completed successfully"
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
    
    # Pause analysis
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

def analyze_transcript_content(transcript_content, age, gender, slp_notes):
    """Analyze transcript content with comprehensive quantification and detailed citations"""
    if not transcript_content or len(transcript_content.strip()) < 50:
        return "Error: Please provide a longer transcript for analysis."
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    # Enhanced comprehensive analysis prompt with detailed quantification
    prompt = f"""
    You are a speech-language pathologist conducting a COMPREHENSIVE speech sample assessment. Provide a SINGLE, DETAILED analysis that quantifies EVERY occurrence and cites specific examples.

    Patient: {age}-year-old {gender}
    
    TRANSCRIPT:
    {transcript_content}{notes_section}
    
    INSTRUCTIONS: Provide ONE comprehensive analysis covering ALL areas below. QUANTIFY EVERYTHING with exact counts and cite SPECIFIC examples from the transcript. Be thorough and detailed. COMPLETE ALL 12 SECTIONS.

    COMPREHENSIVE SPEECH SAMPLE ANALYSIS:

    1. SPEECH FACTORS (with EXACT counts and specific citations):

    A. Fluency Issues:
    - Count and cite EVERY filler word ("um", "uh", "like", "you know", etc.)
    - Count and cite EVERY false start/self-correction
    - Count and cite EVERY repetition of words/phrases
    - Count and cite EVERY revision/restart
    - Calculate percentage of disfluent speech

    B. Word Retrieval Issues:
    - Count and cite EVERY instance of circumlocution
    - Count and cite EVERY incomplete thought/abandoned utterance
    - Count and cite EVERY word-finding pause
    - Count and cite EVERY use of generic terms ("thing", "stuff", etc.)

    C. Grammatical Errors:
    - Count and cite EVERY grammatical error (verb tense, subject-verb agreement, etc.)
    - Count and cite EVERY syntactic error
    - Count and cite EVERY morphological error
    - Count and cite EVERY run-on sentence

    2. LANGUAGE SKILLS ASSESSMENT (with specific evidence):

    A. Lexical/Semantic Skills:
    - Count total unique words vs. total words (Type-Token Ratio)
    - List and categorize vocabulary by sophistication level
    - Identify semantic relationships demonstrated
    - Assess word retrieval strategies used
    - Evaluate semantic precision

    B. Syntactic Skills:
    - Count sentence types (simple, compound, complex, compound-complex)
    - Identify syntactic patterns and errors
    - Assess clause complexity and embedding

    C. Supralinguistic Skills:
    - Identify and cite examples of:
        * Cause-effect relationships
        * Inferences made
        * Non-literal language use
        * Problem-solving language
        * Metalinguistic awareness

    3. COMPLEX SENTENCE ANALYSIS (with exact counts):

    A. Coordinating Conjunctions:
    - Count and cite EVERY use of: and, but, or, so, yet, for, nor
    - Analyze patterns of use
    - Assess age-appropriateness

    B. Subordinating Conjunctions:
    - Count and cite EVERY use of: because, although, while, since, if, when, where, that, which, who, whom, whose
    - Analyze clause complexity
    - Assess embedding depth

    C. Sentence Structure Analysis:
    - Count each sentence type with examples
    - Calculate complexity ratios
    - Assess developmental appropriateness

    4. FIGURATIVE LANGUAGE ANALYSIS (with exact counts):

    A. Similes:
    - Count and cite EVERY simile (comparisons using "like" or "as")
    - Analyze creativity and appropriateness

    B. Metaphors:
    - Count and cite EVERY metaphor (direct comparisons)
    - Assess comprehension and use

    C. Idioms:
    - Count and cite EVERY idiom used
    - Assess comprehension and appropriate use

    D. Non-literal Language:
    - Count and cite EVERY instance of sarcasm, humor, irony
    - Assess comprehension level

    5. PRAGMATIC LANGUAGE ASSESSMENT (with specific examples):

    A. Turn-taking:
    - Analyze conversational flow
    - Count interruptions or overlaps
    - Assess reciprocity

    B. Topic Management:
    - Count topic shifts
    - Assess topic maintenance
    - Evaluate topic introduction

    C. Social Communication:
    - Assess register appropriateness
    - Evaluate politeness markers
    - Analyze social awareness

    6. VOCABULARY AND SEMANTIC ANALYSIS (with quantification):

    A. Vocabulary Diversity:
    - Calculate Type-Token Ratio
    - List most frequent words
    - Assess vocabulary sophistication

    B. Semantic Relationships:
    - Count and cite examples of:
        * Synonyms/antonyms
        * Categories/hierarchies
        * Part-whole relationships
        * Cause-effect vocabulary

    7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS (with counts):

    A. Morphological Markers:
    - Count and cite use of:
        * Plurals (-s, -es)
        * Possessives
        * Verb tenses
        * Derivational morphemes

    B. Phonological Patterns:
    - Identify speech sound errors
    - Count phonological processes
    - Assess syllable structure

    8. COGNITIVE-LINGUISTIC FACTORS (with evidence):

    A. Working Memory:
    - Assess sentence length complexity
    - Analyze information retention
    - Evaluate processing demands

    B. Processing Speed:
    - Note: Processing speed analysis requires audio/video with timestamps
    - If no timing data: analyze linguistic complexity and word-finding patterns
    - Assess cognitive demands of language structures used

    C. Executive Function:
    - Assess planning and organization
    - Evaluate self-monitoring
    - Analyze cognitive flexibility

    9. FLUENCY AND RHYTHM ANALYSIS (with quantification):

    A. Speech Rate:
    - Calculate words per minute (if timing data available)
    - Note: Rate analysis requires audio/video with timestamps
    - If no timing data: analyze word density and complexity patterns

    B. Pause Patterns:
    - Note: Pause analysis requires audio/video with timestamps
    - If no timing data: analyze linguistic markers of hesitation
    - Count disfluencies, fillers, and incomplete utterances

    10. QUANTITATIVE METRICS:

    - Total words: [count]
    - Total utterances: [count]
    - Mean length of utterance (MLU): [calculation]
    - Type-Token Ratio: [calculation]
    - Disfluency rate: [percentage]
    - Error rate: [percentage]
    - Vocabulary diversity score: [calculation]

    11. CLINICAL IMPLICATIONS:

    A. Strengths:
    - List specific strengths with evidence
    - Identify areas of competence

    B. Areas of Need:
    - Identify areas requiring attention
    - Provide specific examples

    C. Assessment Considerations:
    - Note areas that may need formal testing
    - Identify factors requiring further evaluation

    12. PROGNOSIS AND SUMMARY:

    - Overall communication profile
    - Developmental appropriateness
    - Summary of key findings
    - Areas requiring further assessment

    FORMAT REQUIREMENTS:
    - Use bullet points for organization
    - Include exact counts for everything
    - Cite specific quotes from transcript
    - Use clear headings and subheadings
    - Provide percentages and ratios where applicable
    - Be comprehensive but organized
    - Focus on clinical relevance
    - COMPLETE ALL 12 SECTIONS

    MATHEMATICAL ACCURACY REQUIREMENTS:
    - ALL COUNTS MUST MATCH: If you claim "X total", individual counts must sum to X
    - Verify your math before reporting any totals
    - Example: If you find 3 types of errors with counts 5, 7, 2, say "14 total errors" not "12 total errors"
    - Double-check that individual counts sum to your claimed total
    - If counts don't match, correct them before reporting
    - Use precise language: "approximately X" if uncertain, "at least X" for minimum counts
    - This analysis is for students and must be mathematically accurate

    SECTION CHECKLIST - COMPLETE ALL:
    □ 1. SPEECH FACTORS (A, B, C)
    □ 2. LANGUAGE SKILLS ASSESSMENT (A, B, C)
    □ 3. COMPLEX SENTENCE ANALYSIS (A, B, C)
    □ 4. FIGURATIVE LANGUAGE ANALYSIS (A, B, C, D)
    □ 5. PRAGMATIC LANGUAGE ASSESSMENT (A, B, C)
    □ 6. VOCABULARY AND SEMANTIC ANALYSIS (A, B)
    □ 7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS (A, B)
    □ 8. COGNITIVE-LINGUISTIC FACTORS (A, B, C)
    □ 9. FLUENCY AND RHYTHM ANALYSIS (A, B)
    □ 10. QUANTITATIVE METRICS
    □ 11. CLINICAL IMPLICATIONS (A, B, C)
    □ 12. PROGNOSIS AND SUMMARY

    CRITICAL REQUIREMENTS:
    - If you cannot complete all 12 sections in one response, end with <CONTINUE> and continue with the remaining sections
    - Do not skip any sections. Use the checklist to ensure all sections are completed
    - ACCURACY IS PARAMOUNT: This analysis is for students and must be mathematically correct
    - Before reporting any totals, verify that individual counts sum correctly
    - If you find inconsistencies in your counts, resolve them before continuing
    - Use precise language and avoid making claims you cannot support with evidence
    - When in doubt about a count, say "at least X" or "approximately X" rather than claiming certainty
    """
    
    # Get analysis from Claude API
    result = call_claude_api_with_continuation(prompt, max_continuations=5)
    return result

def analyze_transcript(file, age, gender, slp_notes):
    """Analyze transcript from file upload"""
    if file is None:
        return "Please upload a transcript file first."
    
    # Get transcript content
    transcript = process_file(file)
    if transcript.startswith("Error") or transcript.startswith("Please"):
        return transcript
    
    return analyze_transcript_content(transcript, age, gender, slp_notes)

def targeted_analysis(transcript, custom_question, age, gender, slp_notes):
    """Perform targeted analysis based on custom questions with comprehensive detail"""
    if not transcript or not transcript.strip():
        return "Please provide a transcript first."
    
    if not custom_question or not custom_question.strip():
        return "Please enter a specific question for analysis."
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    # Enhanced targeted analysis prompt with comprehensive detail
    prompt = f"""
    You are a speech-language pathologist conducting a DETAILED targeted analysis of a speech transcript.
    
    Patient: {age}-year-old {gender}
    
    TRANSCRIPT:
    {transcript}{notes_section}
    
    SPECIFIC QUESTION FOR ANALYSIS:
    {custom_question.strip()}
    
    INSTRUCTIONS: Provide a COMPREHENSIVE, DETAILED analysis that directly addresses this specific question. Include:
    - EXACT counts and quantification
    - SPECIFIC citations from the transcript
    - DETAILED examples for every observation
    - PERCENTAGES and ratios where applicable
    - CLINICAL significance of findings
    - AGE-APPROPRIATE assessment
    
    ANALYSIS REQUIREMENTS:
    
    1. QUANTIFICATION:
    - Count every relevant occurrence with absolute accuracy
    - Calculate percentages and ratios correctly
    - Provide specific numbers for all observations
    - VERIFY MATH: Ensure individual counts sum to claimed totals
    - Example: If counting 3 types of errors (5, 7, 2), report "14 total errors"
    - Use precise language: "at least X" for minimum counts, "approximately X" if uncertain
    
    2. EVIDENCE:
    - Cite exact quotes from the transcript
    - Provide line-by-line examples
    - Include specific timestamps or context
    
    3. DETAILED EXAMPLES:
    - Give multiple examples for each pattern
    - Show variations in the pattern
    - Demonstrate the range of severity
    
    4. CLINICAL ASSESSMENT:
    - Assess severity level
    - Compare to age expectations
    - Identify clinical significance
    - Suggest intervention implications
    
    5. COMPREHENSIVE COVERAGE:
    - Address all aspects of the question
    - Consider related language areas
    - Include both strengths and weaknesses
    - Provide developmental context
    
    ANALYSIS STRUCTURE:
    
    A. DIRECT ANSWER TO QUESTION:
    - Provide a clear, direct answer
    - Include quantification and severity assessment
    
    B. DETAILED EVIDENCE:
    - List every relevant example with exact quotes
    - Provide counts and percentages
    - Show patterns and variations
    
    C. PATTERN ANALYSIS:
    - Identify underlying patterns
    - Analyze frequency and consistency
    - Assess variability across the transcript
    
    D. DEVELOPMENTAL ASSESSMENT:
    - Compare to age-appropriate expectations
    - Identify developmental level
    - Assess progress and challenges
    
    E. CLINICAL IMPLICATIONS:
    - Impact on communication
    - Effect on academic/social functioning
    - Priority for intervention
    
    F. INTERVENTION CONSIDERATIONS:
    - Specific strategies to address the issue
    - Intensity and frequency recommendations
    - Expected outcomes and timeline
    
    FORMAT REQUIREMENTS:
    - Use clear headings and subheadings
    - Include bullet points for organization
    - Provide exact counts and percentages
    - Cite specific quotes with context
    - Be thorough and comprehensive
    - Focus on clinical relevance and utility
    
    Remember: This should be a DETAILED, COMPREHENSIVE analysis that thoroughly addresses the specific question with quantification, evidence, and clinical implications.
    """
    
    # Get targeted analysis from Claude API
    result = call_claude_api_quick_analysis(prompt)
    return result

# Create enhanced interface with tabs
with gr.Blocks(title="Enhanced Speech Sample Analysis", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🗣️ Enhanced Speech Sample Analysis Tool")
    gr.Markdown("Upload a speech transcript, paste text, or transcribe audio/video and get instant speech sample assessment results with targeted analysis options.")
    
    # Store transcript globally
    transcript_state = gr.State("")
    
    with gr.Tabs():
        # Tab 1: Basic Analysis
        with gr.Tab("�� Basic Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Options")
                    
                    with gr.Tabs():
                        with gr.Tab("📁 File Upload"):
                            file_upload = gr.File(
                                label="Upload Transcript File",
                                file_types=[".txt", ".cha"]
                            )
                            
                            analyze_file_btn = gr.Button(
                                "🔍 Analyze File", 
                                variant="primary"
                            )
                        
                        with gr.Tab("📝 Text Input"):
                            text_input = gr.Textbox(
                                label="Paste Transcript Here",
                                placeholder="Paste your transcript text here...",
                                lines=10
                            )
                            
                            analyze_text_btn = gr.Button(
                                "🔍 Analyze Text", 
                                variant="primary"
                            )
                    
                    gr.Markdown("### Patient Information")
                    
                    age = gr.Number(
                        label="Patient Age", 
                        value=8, 
                        minimum=1, 
                        maximum=120
                    )
                    
                    gender = gr.Radio(
                        ["male", "female", "other"], 
                        label="Gender", 
                        value="male"
                    )
                    
                    slp_notes = gr.Textbox(
                        label="SLP Clinical Notes (Optional)",
                        placeholder="Enter any additional clinical observations, context, or notes...",
                        lines=3
                    )
                
                with gr.Column():
                    gr.Markdown("### Analysis Results")
                    
                    output = gr.Textbox(
                        label="Speech Sample Analysis Report",
                        placeholder="Analysis results will appear here...",
                        lines=25,
                        max_lines=30
                    )
                    
                    analysis_progress = gr.Markdown("")
        
        # Tab 2: Targeted Analysis
        with gr.Tab("🎯 Targeted Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Transcript Input")
                    
                    transcript_input = gr.Textbox(
                        label="Paste Transcript Here",
                        placeholder="Paste your transcript text here, or use the transcript from Basic Analysis...",
                        lines=10
                    )
                    
                    gr.Markdown("### Custom Analysis Question")
                    
                    # Predefined question templates
                    question_templates = gr.Dropdown(
                        choices=[
                            "Select a template or write your own...",
                            "What specific speech patterns indicate word-finding difficulties?",
                            "How does the patient's grammar compare to age expectations?",
                            "What evidence suggests fluency issues in this transcript?",
                            "What pragmatic language skills are demonstrated?",
                            "How does the patient handle complex sentence structures?",
                            "What narrative organization skills are evident?",
                            "What specific intervention targets would you recommend?",
                            "How does this patient's language compare to typical development?",
                            "What evidence suggests cognitive-linguistic strengths/weaknesses?",
                            "Analyze the use of conjunctions and complex sentences",
                            "Identify and analyze figurative language use"
                        ],
                        label="Question Templates (Optional)",
                        value="Select a template or write your own..."
                    )
                    
                    custom_question = gr.Textbox(
                        label="Your Specific Question",
                        placeholder="Enter your specific analysis question here...",
                        lines=3
                    )
                    
                    targeted_analyze_btn = gr.Button(
                        "🎯 Analyze Specific Question", 
                        variant="primary"
                    )
                
                with gr.Column():
                    gr.Markdown("### Targeted Analysis Results")
                    
                    targeted_output = gr.Textbox(
                        label="Targeted Analysis Report",
                        placeholder="Targeted analysis results will appear here...",
                        lines=25,
                        max_lines=30
                    )
                    
                    targeted_progress = gr.Markdown("")
        
        # Tab 3: Quick Questions
        with gr.Tab("⚡ Quick Questions"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Quick Analysis Questions")
                    
                    quick_transcript = gr.Textbox(
                        label="Transcript",
                        placeholder="Paste transcript here...",
                        lines=8
                    )
                    
                    gr.Markdown("### Select Quick Questions")
                    
                    quick_questions = gr.CheckboxGroup(
                        choices=[
                            "Word-finding difficulties",
                            "Grammatical errors",
                            "Fluency issues", 
                            "Pragmatic skills",
                            "Narrative structure",
                            "Vocabulary level",
                            "Complex sentence analysis",
                            "Figurative language use",
                            "Morphological markers",
                            "Phonological patterns",
                            "Cognitive-linguistic factors"
                        ],
                        label="Select questions to analyze:",
                        value=[]
                    )
                    
                    quick_analyze_btn = gr.Button(
                        "⚡ Quick Analysis", 
                        variant="primary"
                    )
                
                with gr.Column():
                    gr.Markdown("### Quick Analysis Results")
                    
                    quick_output = gr.Textbox(
                        label="Quick Analysis Report",
                        placeholder="Quick analysis results will appear here...",
                        lines=25,
                        max_lines=30
                    )
                    
                    quick_progress = gr.Markdown("")
    
    # Event handlers
    def on_analyze_file(file, age_val, gender_val, notes):
        """Handle file analysis"""
        result = analyze_transcript(file, age_val, gender_val, notes)
        transcript = process_file(file) if file else ""
        progress_msg = "✅ Analysis completed" if "[Analysis completed in" in result else "🔄 Analysis in progress..."
        return result, transcript, progress_msg
    
    def on_analyze_text(text, age_val, gender_val, notes):
        """Handle text analysis"""
        result = analyze_transcript_content(text, age_val, gender_val, notes)
        progress_msg = "✅ Analysis completed" if "[Analysis completed in" in result else "🔄 Analysis in progress..."
        return result, text, progress_msg
    
    def on_targeted_analyze(transcript, question, age_val, gender_val, notes):
        """Handle targeted analysis"""
        result = targeted_analysis(transcript, question, age_val, gender_val, notes)
        progress_msg = "✅ Targeted analysis completed" if "[Analysis completed in" in result else "🔄 Targeted analysis in progress..."
        return result, progress_msg
    
    def on_question_template_change(template):
        """Handle question template selection"""
        if template and template != "Select a template or write your own...":
            return template
        return ""
    
    def on_quick_analyze(transcript, questions, age_val, gender_val, notes):
        """Handle quick analysis with multiple questions"""
        if not transcript or not transcript.strip():
            return "Please provide a transcript first.", "❌ No transcript provided"
        
        if not questions:
            return "Please select at least one question to analyze.", "❌ No questions selected"
        
        # Add SLP notes to the prompt if provided
        notes_section = ""
        if notes and notes.strip():
            notes_section = f"""
        
        SLP CLINICAL NOTES:
        {notes.strip()}
        """
        
        # Map selected questions to analysis sections
        question_mapping = {
            "Word-finding difficulties": "WORD-FINDING and RETRIEVAL",
            "Grammatical errors": "GRAMMAR and LANGUAGE ERRORS", 
            "Fluency issues": "FLUENCY and SPEECH RATE",
            "Pragmatic skills": "PRAGMATIC and SOCIAL COMMUNICATION",
            "Narrative structure": "NARRATIVE and DISCOURSE",
            "Vocabulary level": "VOCABULARY and SEMANTIC analysis",
            "Complex sentence analysis": "SYNTAX and COMPLEX SENTENCE analysis",
            "Figurative language use": "FIGURATIVE LANGUAGE analysis",
            "Morphological markers": "MORPHOLOGICAL and PHONOLOGICAL analysis",
            "Phonological patterns": "MORPHOLOGICAL and PHONOLOGICAL analysis",
            "Cognitive-linguistic factors": "COGNITIVE-LINGUISTIC factors"
        }
        
        # Get unique analysis sections for selected questions
        selected_sections = set()
        for question in questions:
            if question in question_mapping:
                selected_sections.add(question_mapping[question])
        
        # Create enhanced quick analysis prompt with comprehensive SLP analysis
        questions_text = "\n".join([f"- {q}" for q in questions])
        sections_text = "\n".join([f"- {section}" for section in selected_sections])
        
        prompt = f"""
        You are a speech-language pathologist conducting a FOCUSED analysis of specific areas in a speech transcript.
        
        Patient: {age_val}-year-old {gender_val}
        
        TRANSCRIPT:
        {transcript}{notes_section}
        
        ANALYZE ONLY THESE SPECIFIC AREAS:
        {questions_text}
        
        ANALYSIS REQUIREMENTS:
        
        For each selected area, provide ONLY:
        1. EXACT COUNTS and quantification
        2. SPECIFIC EXAMPLES with exact quotes from transcript
        3. PERCENTAGES and ratios where applicable
        4. SEVERITY assessment (mild/moderate/severe)
        
        DETAILED ANALYSIS GUIDELINES:
        
        For SYNTAX and COMPLEX SENTENCE analysis:
        - Count and cite EVERY coordinating conjunction (and, but, or, so, yet, for, nor)
        - Count and cite EVERY subordinating conjunction (because, although, while, since, if, when, where, that, which, who, whom, whose)
        - Identify and count each sentence type (simple, compound, complex, compound-complex)
        - Calculate complexity ratios and percentages
        - Assess embedding depth and clause complexity
        - Analyze sentence length patterns and variability
        - Provide specific examples for each pattern
        
        For FIGURATIVE LANGUAGE analysis:
        - Count and cite EVERY simile (comparisons using "like" or "as")
        - Count and cite EVERY metaphor (direct comparisons without "like" or "as")
        - Count and cite EVERY idiom and non-literal expression
        - Provide specific examples with context
        
        For PRAGMATIC and SOCIAL COMMUNICATION:
        - Count and analyze turn-taking patterns (including interruptions, overlaps, response timing)
        - Assess topic maintenance and shifting abilities (including topic introduction, maintenance, and transitions)
        - Evaluate social appropriateness and register use
        - Count interruptions or conversational breakdowns
        - Analyze non-literal language comprehension (sarcasm, humor, irony)
        - Assess conversational repair strategies
        - Evaluate politeness markers and social awareness
        - Provide specific examples of pragmatic behaviors
        
        For VOCABULARY and SEMANTIC analysis:
        - Calculate Type-Token Ratio
        - Count and categorize vocabulary by sophistication level
        - Analyze word retrieval strategies and circumlocution
        - Assess semantic precision and relationships
        - Count academic vs. everyday vocabulary use
        - Analyze vocabulary diversity and range
        - Assess semantic relationships (synonyms, antonyms, categories)
        - Provide specific examples of vocabulary patterns
        
        For MORPHOLOGICAL and PHONOLOGICAL analysis:
        - Count and cite EVERY morphological marker (plurals, possessives, verb tenses)
        - Count and cite EVERY derivational morpheme (prefixes, suffixes)
        - Identify and count phonological patterns and errors
        - Assess syllable structure and stress patterns
        - Analyze morphological complexity
        - Provide specific examples of morphological use
        
        For COGNITIVE-LINGUISTIC factors:
        - Assess working memory demands in language production (sentence length, complexity)
        - Analyze processing speed and efficiency (response time, fluency)
        - Count and evaluate attention and focus patterns
        - Assess executive function skills and self-monitoring (planning, organization, self-correction)
        - Analyze metalinguistic awareness (ability to think about language)
        - Evaluate cognitive flexibility in language use
        - Provide specific examples of cognitive-linguistic patterns
        
        For FLUENCY and SPEECH RATE:
        - Count and cite EVERY disfluency (fillers, repetitions, revisions)
        - Note: Speech rate calculation requires audio/video with timestamps
        - If no timing data: analyze word density and complexity patterns
        - Note: Pause analysis requires audio/video with timestamps
        - If no timing data: analyze linguistic markers of hesitation
        - Count incomplete utterances and word-finding difficulties
        - Provide specific examples of fluency patterns
        
        For GRAMMAR and LANGUAGE ERRORS:
        - Count and cite EVERY grammatical error
        - Count and cite EVERY syntactic error
        - Count and cite EVERY morphological error
        - Calculate error rates and percentages
        - Analyze error patterns and consistency
        - Assess error severity and impact on communication
        - Provide specific examples of error patterns
        
        For WORD-FINDING and RETRIEVAL:
        - Count and cite EVERY instance of circumlocution
        - Count and cite EVERY incomplete thought
        - Count and cite EVERY word-finding pause
        - Analyze word retrieval strategies used
        - Assess naming accuracy and efficiency
        - Evaluate semantic paraphasias and word substitutions
        - Provide specific examples of retrieval patterns
        
        For NARRATIVE and DISCOURSE:
        - Assess narrative organization and coherence
        - Count topic shifts and maintenance
        - Analyze discourse markers and transitions
        - Evaluate story structure and completeness
        - Assess narrative cohesion and logical flow
        - Analyze discourse-level language skills
        - Provide specific examples of narrative patterns
        
        FORMAT REQUIREMENTS:
        - Use clear headings for each area analyzed
        - Include bullet points for organization
        - Provide exact counts and percentages
        - Cite specific quotes from transcript
        - Include severity assessments (mild/moderate/severe) - very simple
        - Focus ONLY on the requested areas
        - Do NOT include age-appropriateness evaluations
        - Do NOT include clinical implications or recommendations
        - Do NOT include intervention suggestions
        - Do NOT include overall assessments
        - Keep analysis focused and concise

        MATHEMATICAL ACCURACY REQUIREMENTS:
        - ALL COUNTS MUST MATCH: Individual counts must sum to claimed totals
        - Verify your math before reporting any numbers
        - Use precise language: "at least X" for minimum counts, "approximately X" if uncertain
        - This analysis is for students and must be mathematically correct
        """
        
        result = call_claude_api_quick_analysis(prompt)
        progress_msg = "✅ Quick analysis completed"
        return result, progress_msg
    
    # Connect event handlers
    analyze_file_btn.click(
        on_analyze_file,
        inputs=[file_upload, age, gender, slp_notes],
        outputs=[output, transcript_input, analysis_progress]
    )
    
    analyze_text_btn.click(
        on_analyze_text,
        inputs=[text_input, age, gender, slp_notes],
        outputs=[output, transcript_input, analysis_progress]
    )
    
    targeted_analyze_btn.click(
        on_targeted_analyze,
        inputs=[transcript_input, custom_question, age, gender, slp_notes],
        outputs=[targeted_output, targeted_progress]
    )
    
    question_templates.change(
        on_question_template_change,
        inputs=[question_templates],
        outputs=[custom_question]
    )
    
    quick_analyze_btn.click(
        on_quick_analyze,
        inputs=[quick_transcript, quick_questions, age, gender, slp_notes],
        outputs=[quick_output, quick_progress]
    )

if __name__ == "__main__":
    print("🚀 Starting Enhanced Speech Sample Analysis Tool...")
    print("📊 Features: Basic Analysis, Targeted Questions, Quick Multi-Analysis")
    print("📈 Analysis: Complex sentences, figurative language, pragmatic skills, cognitive-linguistic factors")
    
    if not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not configured - analysis will show error message")
        print("   For HuggingFace Spaces: Add ANTHROPIC_API_KEY as a secret in your space settings")
        print("   For local use: export ANTHROPIC_API_KEY='your-key-here'")
    else:
        print("✅ Claude API configured")
    
    if not SPEECHBRAIN_AVAILABLE:
        print("⚠️  SpeechBrain not available - transcription will be disabled")
        print("   Install with: pip install speechbrain transformers torch")
    else:
        print("✅ SpeechBrain available for transcription")
    
    if not MOVIEPY_AVAILABLE:
        print("⚠️  MoviePy not available - video processing will be limited")
        print("   Install with: pip install moviepy")
    else:
        print("✅ MoviePy available for video processing")
    
    if not DIARIZATION_AVAILABLE:
        print("⚠️  Pyannote.audio not available - speaker diarization will be disabled")
        print("   Install with: pip install pyannote.audio")
        print("   Note: Requires HuggingFace token for model access")
    else:
        print("✅ Pyannote.audio available for speaker diarization")
    
    if not SENTIMENT_AVAILABLE:
        print("⚠️  Transformers not available - sentiment/emotion analysis will be disabled")
        print("   Install with: pip install transformers torch")
    else:
        print("✅ Transformers available for sentiment and emotion analysis")
    
    app.launch(show_api=False)