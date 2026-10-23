import gradio as gr
import json
import os
import logging
import requests
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anthropic API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

if ANTHROPIC_API_KEY:
    logger.info("Claude API key found")
else:
    logger.warning("Claude API key not found - using demo mode")

def clean_output_formatting(text):
    """Remove asterisks, hashtags, and convert tables to lists in NLP section"""
    import re
    
    # Remove all asterisks (bolding)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *text* -> text
    text = text.replace('**', '')                    # Remove any remaining **
    text = text.replace('*', '')                     # Remove any remaining *
    
    # Remove hashtags from headers (at start of line)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Remove ### headers
    
    # Also remove hashtags that appear mid-sentence or in other positions
    text = re.sub(r'\s#{1,6}\s+', ' ', text)  # Remove hashtags with spaces around them
    text = re.sub(r'#{1,6}([A-Z])', r'\1', text)  # Remove # before capitalized words like #SECTION
    text = re.sub(r'^#{1,6}$', '', text, flags=re.MULTILINE)  # Remove standalone hashtags on a line
    text = re.sub(r'#([a-zA-Z])', r'\1', text)  # Remove # before any word
    text = text.replace('#', '')  # Remove any remaining bare # symbols
    
    # Convert tables to lists - more comprehensive approach
    lines = text.split('\n')
    cleaned_lines = []
    in_table = False
    
    for line in lines:
        # Detect table start (line with multiple |)
        if line.count('|') >= 2 and not in_table:
            in_table = True
            # Skip header line, will process data rows
            continue
        elif line.count('|') >= 2 and in_table:
            # This is a table row - convert to bullet point
            if not re.match(r'^\s*\|[\s\-\|]+\|\s*$', line):  # Skip separator lines
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) >= 2:
                    cleaned_lines.append(f"- {cells[0]}: {' '.join(cells[1:])}")
        elif in_table and line.count('|') < 2:
            # End of table
            in_table = False
            cleaned_lines.append(line)
        else:
            # Regular line
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text

def segment_response_by_sections(response_text):
    """Segment response by section titles and return a dictionary of sections"""
    required_sections = [
        "1. SPEECH FACTORS",
        "2. LANGUAGE SKILLS ASSESSMENT", 
        "3. COMPLEX SENTENCE ANALYSIS",
        "4. FIGURATIVE LANGUAGE ANALYSIS",
        "5. PRAGMATIC LANGUAGE ASSESSMENT",
        "6. VOCABULARY AND SEMANTIC ANALYSIS",
        "7. NLP-DERIVED LINGUISTIC FEATURES",
        "8. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS",
        "9. COGNITIVE-LINGUISTIC FACTORS",
        "10. FLUENCY AND RHYTHM ANALYSIS",
        "11. QUANTITATIVE METRICS",
        "12. CLINICAL IMPLICATIONS",
        "13. PROGNOSIS AND SUMMARY"
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
        "8. QUANTITATIVE METRICS AND NLP FEATURES"
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


def call_claude_api_quick_analysis(prompt):
    """Call Claude API for quick focused analysis - single response only"""
    if not ANTHROPIC_API_KEY:
        return "Error: Claude API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-sonnet-4-5",
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
            timeout=180
        )
        
        if response.status_code == 200:
            response_json = response.json()
            raw_response = response_json['content'][0]['text']
            # Clean formatting from response (removes **, #, tables)
            cleaned_response = clean_output_formatting(raw_response)
            return cleaned_response
        else:
            logger.error(f"Claude API error: {response.status_code} - {response.text}")
            return f"Error: Claude API Error: {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return f"Error: {str(e)}"

def call_claude_api(prompt):
    """Call Claude API directly (legacy function for backward compatibility)"""
    return call_claude_api_quick_analysis(prompt)

def answer_quick_question(transcript_content, question, age, gender, slp_notes):
    """Answer a specific question about the transcript quickly using annotated version for accuracy"""
    if not transcript_content or len(transcript_content.strip()) < 20:
        return "Error: Please provide a transcript for analysis."
    
    if not question or len(question.strip()) < 5:
        return "Error: Please provide a specific question."
    
    # First, annotate the transcript to get accurate markers
    logger.info("Annotating transcript for accurate question answering...")
    annotated_transcript = annotate_transcript(transcript_content, age, gender, slp_notes)
    
    # Check if annotation was successful
    if annotated_transcript.startswith("Error") or annotated_transcript.startswith("ANNOTATION INCOMPLETE"):
        logger.warning("Using original transcript due to annotation issues")
        analysis_transcript = transcript_content
    else:
        analysis_transcript = annotated_transcript
    
    # Calculate linguistic metrics and lexical diversity
    logger.info("Calculating linguistic metrics and lexical diversity...")
    linguistic_metrics = calculate_linguistic_metrics(analysis_transcript)
    lexical_diversity = calculate_advanced_lexical_diversity(analysis_transcript)
    marker_analysis = analyze_annotation_markers(analysis_transcript)
    
    # Build metrics section for the prompt
    metrics_section = "\n\nLINGUISTIC METRICS FOR REFERENCE:\n"
    metrics_section += f"Total words: {linguistic_metrics.get('total_words', 0)}\n"
    metrics_section += f"Total sentences: {linguistic_metrics.get('total_sentences', 0)}\n"
    metrics_section += f"Unique words: {linguistic_metrics.get('unique_words', 0)}\n"
    metrics_section += f"Type-Token Ratio (TTR): {linguistic_metrics.get('type_token_ratio', 0):.3f}\n"
    metrics_section += f"Mean Length of Utterance (words): {linguistic_metrics.get('mlu_words', 0):.2f}\n"
    
    if lexical_diversity.get('library_available', False) and 'diversity_measures' in lexical_diversity:
        measures = lexical_diversity['diversity_measures']
        metrics_section += "\nAdvanced Lexical Diversity Measures:\n"
        if measures.get('root_ttr') is not None:
            metrics_section += f"Root TTR: {measures['root_ttr']:.4f}\n"
        if measures.get('hdd') is not None:
            metrics_section += f"HDD: {measures['hdd']:.4f}\n"
        if measures.get('mtld') is not None:
            metrics_section += f"MTLD: {measures['mtld']:.4f}\n"
    
    marker_counts = marker_analysis.get('marker_counts', {})
    if any(marker_counts.values()):
        metrics_section += "\nAnnotation Marker Summary:\n"
        metrics_section += f"Total fluency issues: {marker_analysis['category_totals'].get('fluency_issues', 0)}\n"
        metrics_section += f"Total grammar errors: {marker_analysis['category_totals'].get('grammar_errors', 0)}\n"
        metrics_section += f"Simple vocabulary: {marker_analysis['category_totals'].get('simple_vocabulary', 0)}\n"
        metrics_section += f"Complex vocabulary: {marker_analysis['category_totals'].get('complex_vocabulary', 0)}\n"
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    prompt = f"""
    You are a speech-language pathologist answering a specific question about a speech sample.
    
    TRANSCRIPT (with annotation markers for reference):
    {analysis_transcript}{notes_section}{metrics_section}
    
    ANNOTATION MARKER REFERENCE:
    FLUENCY: [FILLER], [FALSE_START], [REPETITION], [REVISION], [PAUSE]
    WORD RETRIEVAL: [CIRCUMLOCUTION], [INCOMPLETE], [GENERIC], [WORD_SEARCH]
    GRAMMAR: [GRAM_ERROR], [SYNTAX_ERROR], [MORPH_ERROR], [RUN_ON]
    VOCABULARY: [SIMPLE_VOCAB], [COMPLEX_VOCAB], [SEMANTIC_ERROR]
    PRAGMATICS: [TOPIC_SHIFT], [TANGENT], [INAPPROPRIATE], [COHERENCE_BREAK]
    SENTENCE STRUCTURE: [SIMPLE_SENT], [COMPLEX_SENT], [COMPOUND_SENT], [FIGURATIVE]
    OTHER: [PRONOUN_REF], [MAZING], [PERSEVERATION]
    
    QUESTION: {question}
    
    INSTRUCTIONS:
    Provide a focused, detailed answer to the specific question asked
    Include specific examples from the transcript with exact quotes
    Use the annotation markers to identify and count specific features accurately
    Incorporate the provided linguistic metrics and lexical diversity measures when relevant
    Provide quantitative data when relevant (counts, percentages, rates)
    Provide objective data interpretation only
    Keep the response focused on the question but thorough in analysis
    If the question relates to multiple areas, address all relevant aspects
    Do NOT use asterisks (**), hashtags (#), or bold formatting in your response. Use plain text only.
    
    Answer the question with specific evidence from the transcript:
    """
    
    return call_claude_api_quick_analysis(prompt)

def analyze_targeted_area(transcript_content, analysis_area, age, gender, slp_notes):
    """Perform targeted analysis of a specific area using annotated transcript for accuracy"""
    if not transcript_content or len(transcript_content.strip()) < 20:
        return "Error: Please provide a transcript for analysis."
    
    # First, annotate the transcript to get accurate markers
    logger.info("Annotating transcript for accurate analysis...")
    annotated_transcript = annotate_transcript(transcript_content, age, gender, slp_notes)
    
    # Check if annotation was successful
    if annotated_transcript.startswith("Error") or annotated_transcript.startswith("ANNOTATION INCOMPLETE"):
        logger.warning("Using original transcript due to annotation issues")
        analysis_transcript = transcript_content
    else:
        analysis_transcript = annotated_transcript
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    # Define analysis prompts for different areas
    analysis_prompts = {
        "Fluency and Disfluencies": """
        Conduct a comprehensive FLUENCY ANALYSIS using annotation markers as reference.
        
        TRANSCRIPT WITH ANNOTATION MARKERS:
        {analysis_transcript}{notes_section}
        
        MARKER REFERENCE:
        [FILLER] = filler words (um, uh, like, you know, etc.)
        [FALSE_START] = false starts and self-corrections
        [REPETITION] = word/phrase repetitions
        [REVISION] = revisions and restarts
        [PAUSE] = hesitations and pauses
        
        1. DISFLUENCY TYPES AND COUNTS:
        Count each marker type precisely from the annotated transcript
        Provide exact quotes showing each marker
        Calculate rates per 100 words
        
        2. DISFLUENCY PATTERNS:
        Identify most frequent disfluency types by count
        Analyze clustering patterns where disfluencies concentrate
        Assess impact on communication effectiveness
        
        3. FLUENCY FACILITATORS:
        Identify fluent segments with no markers
        Note contexts that show high fluency
        Assess overall speech rhythm and flow
        
        4. OBJECTIVE SUMMARY:
        Provide data summary with counts and rates
        List observed patterns only
        
        Do NOT use asterisks (**), hashtags (#), or bold formatting. Use plain text only.
        """,
        
        "Grammar and Syntax": """
        Conduct a comprehensive GRAMMATICAL ANALYSIS using annotation markers as reference.
        
        TRANSCRIPT WITH ANNOTATION MARKERS:
        {analysis_transcript}{notes_section}
        
        MARKER REFERENCE:
        [GRAM_ERROR] = grammatical errors
        [SYNTAX_ERROR] = word order/syntax problems
        [MORPH_ERROR] = morphological errors (plurals, tense, etc.)
        [RUN_ON] = run-on sentences
        
        1. MORPHOLOGICAL ANALYSIS:
        Count [MORPH_ERROR] markers and categorize by type
        Identify patterns in morphological errors
        Analyze error frequency
        
        2. SYNTACTIC STRUCTURES:
        Analyze [SIMPLE_SENT], [COMPOUND_SENT], [COMPLEX_SENT] markers
        Count [SYNTAX_ERROR] and [GRAM_ERROR] markers
        Assess word order patterns
        Evaluate conjunction and subordination use
        
        3. VERB USAGE:
        Identify [GRAM_ERROR] markers related to verbs
        Analyze tense consistency
        Count subject-verb agreement errors
        
        4. OBJECTIVE SUMMARY:
        List primary grammatical patterns observed
        Provide count data only
        
        Do NOT use asterisks (**), hashtags (#), or bold formatting. Use plain text only.
        """,
        
        "Vocabulary and Semantics": """
        Conduct a comprehensive VOCABULARY ANALYSIS using annotation markers as reference.
        
        TRANSCRIPT WITH ANNOTATION MARKERS:
        {analysis_transcript}{notes_section}
        
        MARKER REFERENCE:
        [SIMPLE_VOCAB] = basic/high-frequency words
        [COMPLEX_VOCAB] = sophisticated/low-frequency words
        [SEMANTIC_ERROR] = inappropriate word choices
        [GENERIC] = vague terms (thing, stuff)
        [CIRCUMLOCUTION] = roundabout descriptions
        [WORD_SEARCH] = explicit word-finding attempts
        
        1. LEXICAL DIVERSITY:
        Count [SIMPLE_VOCAB] and [COMPLEX_VOCAB] markers
        Calculate vocabulary sophistication ratio
        Identify vocabulary diversity patterns
        
        2. SEMANTIC ACCURACY:
        Count [SEMANTIC_ERROR] markers with quotes
        Identify [GENERIC] term usage
        Count [CIRCUMLOCUTION] and [WORD_SEARCH] markers
        Assess word precision
        
        3. VOCABULARY CATEGORIES:
        Analyze patterns in vocabulary type markers
        Identify high-frequency vs. low-frequency word usage
        Assess conversational vs. academic vocabulary
        
        4. WORD RETRIEVAL:
        Count word-finding difficulties [WORD_SEARCH], [CIRCUMLOCUTION], [GENERIC]
        Identify compensatory strategies
        Assess retrieval efficiency by frequency
        
        5. OBJECTIVE SUMMARY:
        List vocabulary patterns observed with counts
        Provide data summary only
        
        Do NOT use asterisks (**), hashtags (#), or bold formatting. Use plain text only.
        """,
        
        "Pragmatics and Discourse": """
        Conduct a comprehensive PRAGMATIC ANALYSIS using annotation markers as reference.
        
        TRANSCRIPT WITH ANNOTATION MARKERS:
        {analysis_transcript}{notes_section}
        
        MARKER REFERENCE:
        [TOPIC_SHIFT] = topic changes
        [TANGENT] = tangential/off-topic speech
        [INAPPROPRIATE] = inappropriate content
        [COHERENCE_BREAK] = incoherent statements
        [PRONOUN_REF] = unclear pronoun references
        
        1. DISCOURSE ORGANIZATION:
        Count [TOPIC_SHIFT] and [TANGENT] markers
        Assess narrative structure and coherence
        Evaluate logical idea sequencing
        
        2. CONVERSATIONAL SKILLS:
        Analyze topic maintenance between [TOPIC_SHIFT] markers
        Assess response appropriateness
        Evaluate communication effectiveness
        
        3. REFERENTIAL COMMUNICATION:
        Count [PRONOUN_REF] markers
        Assess clarity of pronoun use
        Evaluate referential precision
        
        4. PRAGMATIC APPROPRIATENESS:
        Count [INAPPROPRIATE] markers if present
        Assess contextual appropriateness of content
        Evaluate social communication awareness
        
        5. OBJECTIVE SUMMARY:
        List pragmatic patterns observed with marker counts
        Provide data summary only
        
        Do NOT use asterisks (**), hashtags (#), or bold formatting. Use plain text only.
        """,
        
        "Sentence Complexity": """
        Conduct a comprehensive SENTENCE COMPLEXITY ANALYSIS using annotation markers as reference.
        
        TRANSCRIPT WITH ANNOTATION MARKERS:
        {analysis_transcript}{notes_section}
        
        MARKER REFERENCE:
        [SIMPLE_SENT] = simple sentences
        [COMPOUND_SENT] = compound sentences
        [COMPLEX_SENT] = complex sentences
        [FIGURATIVE] = figurative language/idioms
        
        1. SENTENCE TYPES:
        Count [SIMPLE_SENT], [COMPOUND_SENT], [COMPLEX_SENT] markers
        Calculate percentage distribution of each type
        Provide examples of each type
        
        2. CLAUSE ANALYSIS:
        Analyze clause density from complex sentence markers
        Count subordinate and coordinate clause patterns
        Assess coordination and subordination use
        
        3. PHRASE STRUCTURES:
        Analyze complexity patterns within sentence markers
        Assess phrase elaboration levels
        Evaluate prepositional phrase usage
        
        4. SYNTACTIC MATURITY:
        Calculate Mean Length of Utterance (MLU) from sentence length patterns
        List syntactic patterns observed
        
        5. OBJECTIVE SUMMARY:
        Provide complexity data with counts and percentages
        List observed patterns only
        
        Do NOT use asterisks (**), hashtags (#), or bold formatting. Use plain text only.
        """,
        
        "Word Finding and Retrieval": """
        Conduct a comprehensive WORD RETRIEVAL ANALYSIS using annotation markers as reference.
        
        TRANSCRIPT WITH ANNOTATION MARKERS:
        {analysis_transcript}{notes_section}
        
        MARKER REFERENCE:
        [WORD_SEARCH] = explicit word-finding attempts
        [CIRCUMLOCUTION] = roundabout descriptions as workarounds
        [GENERIC] = vague terms (thing, stuff, whatsit)
        [INCOMPLETE] = abandoned thoughts/word-finding failures
        
        1. WORD-FINDING DIFFICULTIES:
        Count [WORD_SEARCH], [CIRCUMLOCUTION], [GENERIC], [INCOMPLETE] markers
        Provide exact quotes showing each type
        Calculate frequency of each difficulty type
        
        2. RETRIEVAL STRATEGIES:
        Identify compensatory strategies from marker patterns
        Analyze self-cueing attempts marked with [WORD_SEARCH]
        Assess success rate from [CIRCUMLOCUTION] effectiveness
        
        3. ERROR PATTERNS:
        Categorize word-finding issues by marker type
        Identify semantic vs. phonological retrieval issues
        Analyze error consistency patterns
        
        4. CONTEXTUAL FACTORS:
        Identify contexts that show high [WORD_SEARCH] density
        Assess topic complexity impact on word retrieval
        Evaluate linguistic complexity effects
        
        5. OBJECTIVE SUMMARY:
        List word-finding patterns observed with marker counts
        Provide data summary only
        
        Do NOT use asterisks (**), hashtags (#), or bold formatting. Use plain text only.
        """
    }
    
    if analysis_area not in analysis_prompts:
        return f"Error: Analysis area '{analysis_area}' not recognized. Available areas: {', '.join(analysis_prompts.keys())}"
    
    # Get the base prompt and insert the analysis transcript
    base_prompt_template = analysis_prompts[analysis_area]
    base_prompt = base_prompt_template.format(analysis_transcript=analysis_transcript, notes_section=notes_section)
    
    # Calculate linguistic metrics and lexical diversity
    logger.info("Calculating linguistic metrics and lexical diversity...")
    linguistic_metrics = calculate_linguistic_metrics(analysis_transcript)
    lexical_diversity = calculate_advanced_lexical_diversity(analysis_transcript)
    marker_analysis = analyze_annotation_markers(analysis_transcript)
    
    # Build metrics section for the prompt
    metrics_section = "\n\nLINGUISTIC METRICS FOR REFERENCE:\n"
    metrics_section += f"Total words: {linguistic_metrics.get('total_words', 0)}\n"
    metrics_section += f"Total sentences: {linguistic_metrics.get('total_sentences', 0)}\n"
    metrics_section += f"Unique words: {linguistic_metrics.get('unique_words', 0)}\n"
    metrics_section += f"Type-Token Ratio (TTR): {linguistic_metrics.get('type_token_ratio', 0):.3f}\n"
    metrics_section += f"Mean Length of Utterance (words): {linguistic_metrics.get('mlu_words', 0):.2f}\n"
    metrics_section += f"Mean Length of Utterance (morphemes): {linguistic_metrics.get('mlu_morphemes', 0):.2f}\n"
    
    if lexical_diversity.get('library_available', False) and 'diversity_measures' in lexical_diversity:
        measures = lexical_diversity['diversity_measures']
        metrics_section += "\nAdvanced Lexical Diversity Measures:\n"
        if measures.get('root_ttr') is not None:
            metrics_section += f"Root TTR: {measures['root_ttr']:.4f}\n"
        if measures.get('log_ttr') is not None:
            metrics_section += f"Log TTR: {measures['log_ttr']:.4f}\n"
        if measures.get('hdd') is not None:
            metrics_section += f"HDD: {measures['hdd']:.4f}\n"
        if measures.get('mtld') is not None:
            metrics_section += f"MTLD: {measures['mtld']:.4f}\n"
    
    marker_counts = marker_analysis.get('marker_counts', {})
    if any(marker_counts.values()):
        metrics_section += "\nAnnotation Marker Summary:\n"
        metrics_section += f"Total fluency issues: {marker_analysis['category_totals'].get('fluency_issues', 0)}\n"
        metrics_section += f"Total grammar errors: {marker_analysis['category_totals'].get('grammar_errors', 0)}\n"
        metrics_section += f"Simple vocabulary: {marker_analysis['category_totals'].get('simple_vocabulary', 0)}\n"
        metrics_section += f"Complex vocabulary: {marker_analysis['category_totals'].get('complex_vocabulary', 0)}\n"
        vocab_ratio = marker_analysis['category_totals'].get('vocab_sophistication_ratio', 0)
        if vocab_ratio > 0:
            metrics_section += f"Vocabulary sophistication ratio: {vocab_ratio:.3f}\n"
    
    prompt = f"""
    You are a speech-language pathologist conducting a targeted analysis of a specific area.
    
    ANALYSIS FOCUS: {analysis_area}
    
    {base_prompt}{metrics_section}
    
    INSTRUCTIONS:
    Provide specific examples with exact quotes from the transcript
    Include quantitative data using marker counts and percentages
    Incorporate the provided linguistic metrics and lexical diversity measures when relevant
    Provide objective data interpretation only
    Focus on measurable observations
    Be thorough but focused on the specified area
    
    Conduct the targeted analysis:
    """
    
    return call_claude_api_quick_analysis(prompt)

def check_annotation_completeness(original_transcript, annotated_transcript):
    """Check if annotation is complete by verifying last 3 words are present"""
    import re
    
    # Clean and extract words from original transcript
    original_words = re.findall(r'\b\w+\b', original_transcript.strip())
    if len(original_words) < 3:
        return True, "Transcript too short to validate"
    
    # Get last 3 words from original
    last_three_words = original_words[-3:]
    
    # Clean annotated transcript (remove markers but keep words)
    cleaned_annotated = re.sub(r'\[.*?\]', '', annotated_transcript)
    annotated_words = re.findall(r'\b\w+\b', cleaned_annotated.strip())
    
    # Check if all last 3 words appear in the annotated transcript
    missing_words = []
    for word in last_three_words:
        if word.lower() not in [w.lower() for w in annotated_words]:
            missing_words.append(word)
    
    if missing_words:
        return False, f"Annotation appears incomplete. Missing words from end: {', '.join(missing_words)}"
    
    # Additional check: verify the last few words appear near the end
    if len(annotated_words) > 0:
        last_annotated_words = annotated_words[-10:]  # Check last 10 words
        last_original_in_annotated = sum(1 for word in last_three_words 
                                       if word.lower() in [w.lower() for w in last_annotated_words])
        
        if last_original_in_annotated < 2:  # At least 2 of the last 3 should be near the end
            return False, f"Annotation may be incomplete. Last words '{', '.join(last_three_words)}' not found near end of annotation"
    
    return True, "Annotation appears complete"

def annotate_transcript(transcript_content, age, gender, slp_notes):
    """First step: Annotate transcript with linguistic markers"""
    if not transcript_content or len(transcript_content.strip()) < 50:
        return "Error: Please provide a longer transcript for annotation."
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    annotation_prompt = f"""
    You are a speech-language pathologist preparing a transcript for detailed analysis. Your task is to ANNOTATE the ENTIRE transcript with linguistic markers at a WORD-BY-WORD level.
    
    ORIGINAL TRANSCRIPT:
    {transcript_content}{notes_section}
    
    CRITICAL REQUIREMENT: You MUST annotate the COMPLETE transcript. Do NOT provide partial annotations or stop mid-sentence. Complete the ENTIRE transcript annotation in one response.

    DETAILED ANNOTATION INSTRUCTIONS:
    Annotate by adding markers in brackets IMMEDIATELY after each relevant word or phrase:

    FLUENCY MARKERS:
    - [FILLER] after: um[FILLER], uh[FILLER], like[FILLER], you know[FILLER], well[FILLER], so[FILLER]
    - [FALSE_START] after incomplete words: "I was go-[FALSE_START] going"
    - [REPETITION] after repeated words: "I I[REPETITION] went"
    - [REVISION] after self-corrections: "I went to the-[REVISION] I mean"
    - [PAUSE] for hesitations: "I was...[PAUSE] thinking"

    WORD RETRIEVAL MARKERS:
    - [CIRCUMLOCUTION] after roundabout descriptions: "that thing you write with[CIRCUMLOCUTION]"
    - [INCOMPLETE] after abandoned thoughts: "I was thinking about the...[INCOMPLETE]"
    - [GENERIC] after vague terms: thing[GENERIC], stuff[GENERIC], whatsit[GENERIC]
    - [WORD_SEARCH] after searching: "the... um...[WORD_SEARCH] car"

    GRAMMATICAL MARKERS:
    - [GRAM_ERROR] after mistakes: "I goed[GRAM_ERROR]", "He don't[GRAM_ERROR]"
    - [SYNTAX_ERROR] after word order problems: "Yesterday I to store went[SYNTAX_ERROR]"
    - [MORPH_ERROR] after morphological errors: "runned[MORPH_ERROR]", "childs[MORPH_ERROR]"
    - [RUN_ON] at end of run-on sentences

    VOCABULARY MARKERS:
    - [SIMPLE_VOCAB] after basic words: go[SIMPLE_VOCAB], big[SIMPLE_VOCAB], good[SIMPLE_VOCAB]
    - [COMPLEX_VOCAB] after sophisticated words: magnificent[COMPLEX_VOCAB], elaborate[COMPLEX_VOCAB]
    - [SEMANTIC_ERROR] after wrong word choices: "drove my bicycle[SEMANTIC_ERROR]"

    PRAGMATIC MARKERS:
    - [TOPIC_SHIFT] after topic changes: "Anyway, about cats[TOPIC_SHIFT]"
    - [TANGENT] after going off-topic: "Speaking of dogs, my vacation[TANGENT]"
    - [INAPPROPRIATE] after inappropriate content
    - [COHERENCE_BREAK] after illogical statements

    SENTENCE COMPLEXITY MARKERS:
    - [SIMPLE_SENT] after simple sentences: "I went home.[SIMPLE_SENT]"
    - [COMPLEX_SENT] after complex sentences: "When I got home, I made dinner.[COMPLEX_SENT]"
    - [COMPOUND_SENT] after compound sentences: "I went home, and made dinner.[COMPOUND_SENT]"
    - [FIGURATIVE] after metaphors/idioms: "raining cats and dogs[FIGURATIVE]"

    ADDITIONAL MARKERS:
    - [PRONOUN_REF] after unclear pronouns: "He told him that he[PRONOUN_REF] was wrong"
    - [MAZING] after confusing constructions
    - [PERSEVERATION] after repetitive patterns

    MANDATORY REQUIREMENTS:
    1. Do NOT stop until the entire transcript is complete
    2. Keep ALL original text intact
    3. Mark overlapping features when applicable
    4. Be consistent throughout
    5. Complete the annotation in ONE response - no partial outputs allowed

    PROVIDE THE COMPLETE ANNOTATED TRANSCRIPT - EVERY WORD MUST BE PROCESSED.
    """
    
    # Get initial annotation
    annotated_result = call_claude_api(annotation_prompt)
    
    # Check if annotation is complete
    is_complete, validation_message = check_annotation_completeness(transcript_content, annotated_result)
    
    if not is_complete:
        logger.warning(f"Annotation incomplete: {validation_message}")
        
        # Try once more with stronger emphasis on completion
        retry_prompt = f"""
        CRITICAL: The previous annotation was INCOMPLETE. You MUST complete the ENTIRE transcript.
        
        {validation_message}
        
        ORIGINAL TRANSCRIPT (COMPLETE THIS ENTIRELY):
        {transcript_content}{notes_section}
        
        MANDATORY REQUIREMENT: Annotate EVERY SINGLE WORD from start to finish. Do not stop until you reach the very last word of the transcript.
        
        {annotation_prompt.split('DETAILED ANNOTATION INSTRUCTIONS:')[1]}
        
        VERIFY: The last words of the original transcript are: {' '.join(transcript_content.strip().split()[-3:])}
        ENSURE these words appear at the END of your annotated transcript.
        """
        
        retry_result = call_claude_api(retry_prompt)
        
        # Check retry
        retry_complete, retry_message = check_annotation_completeness(transcript_content, retry_result)
        
        if retry_complete:
            logger.info("Retry successful - annotation now complete")
            return retry_result
        else:
            logger.error(f"Retry failed: {retry_message}")
            return f"ANNOTATION INCOMPLETE: {retry_message}\n\nPartial annotation:\n{retry_result}"
    
    logger.info("Annotation completed successfully")
    return annotated_result

def analyze_annotated_transcript(annotated_transcript, age, gender, slp_notes):
    """Second step: Analyze the annotated transcript with comprehensive quantification"""
    if not annotated_transcript or len(annotated_transcript.strip()) < 50:
        return "Error: Please provide an annotated transcript for analysis."
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    analysis_prompt = f"""
    You are a speech-language pathologist conducting a comprehensive analysis of an annotated speech sample. Provide objective data analysis without clinical interpretations.

    ANNOTATED TRANSCRIPT:
    {annotated_transcript}{notes_section}
    
    INSTRUCTIONS: Complete ALL 8 sections below. Use simple formatting with NO BOLDING (no ** or asterisks), NO hashtags (###), and minimal markdown. Focus on objective data only. Count all markers precisely and provide specific examples. Write section headers as plain text followed by a colon. DO NOT include age/gender comparisons, clinical interpretations, severity assessments, or treatment recommendations.

    COMPREHENSIVE SPEECH SAMPLE ANALYSIS

    1. SPEECH FACTORS

    A. Fluency Issues (count each marker type precisely):
    - Filler words ([FILLER]): Count all instances, calculate rate per 100 words
      * List each type: "um," "uh," "like," "you know," etc.
      * Provide specific examples with context
      * Calculate percentage of total words
    - False starts ([FALSE_START]): Count and categorize
      * Word-level false starts: "I was go- going"
      * Phrase-level false starts: "My bike is- I mean my bike looks"
      * Provide exact quotes from transcript
    - Repetitions ([REPETITION]): Count and categorize by type
      * Word repetitions: "I I went"
      * Phrase repetitions: "to the store to the store"
      * Sound repetitions: "b-b-bike"
    - Revisions ([REVISION]): Count self-corrections and analyze patterns
      * Grammatical revisions: "I goed- I went"
      * Lexical revisions: "big- huge dog"
      * Semantic revisions: "car- I mean bike"
    - Pauses ([PAUSE]): Count hesitation markers and silent pauses
    - Total disfluency rate: Calculate combined rate per 100 words

    B. Word Retrieval Issues (detailed analysis):
    - Circumlocutions ([CIRCUMLOCUTION]): Count and analyze strategies
      * Functional descriptions: "the thing you write with"
      * Category + description: "that type of fish in the salad"
      * Provide exact quotes and analyze effectiveness
    - Incomplete thoughts ([INCOMPLETE]): Count abandoned utterances
      * Analyze patterns: topic-related, complexity-related, retrieval-related
    - Generic terms ([GENERIC]): Count vague language
      * "thing," "stuff," "something," "whatsit"
      * Calculate specificity ratio
    - Word searches ([WORD_SEARCH]): Count explicit retrieval attempts
      * "What do you call it," "I can't think of the word"
    - Overall efficiency: Calculate success rate of retrieval attempts

    C. Grammatical Errors (comprehensive breakdown):
    - Grammatical errors ([GRAM_ERROR]): Count by subcategory
      * Subject-verb agreement: "He don't like it"
      * Verb tense errors: "Yesterday I go to store"
      * Pronoun errors: "Me and him went"
      * Article errors: "I saw a elephant"
    - Syntax errors ([SYNTAX_ERROR]): Count word order problems
    - Morphological errors ([MORPH_ERROR]): Count and categorize
      * Plural errors: "childs," "foots"
      * Past tense errors: "runned," "catched"
      * Comparative errors: "more better"
    - Run-on sentences ([RUN_ON]): Count and assess boundary awareness
    - Calculate grammatical accuracy rate

    2. LANGUAGE SKILLS ASSESSMENT

    A. Vocabulary Analysis (detailed breakdown):
    - Simple vocabulary ([SIMPLE_VOCAB]): Count and categorize
      * High-frequency words: "go," "big," "good"
      * Basic descriptors: "nice," "fun," "cool"
      * Calculate percentage of total vocabulary
    - Complex vocabulary ([COMPLEX_VOCAB]): Count and analyze
      * Academic vocabulary: "magnificent," "elaborate"
      * Technical terms: "carburetor," "photosynthesis"
      * Low-frequency words: "churrasco," "anchovies"
    - Vocabulary sophistication ratio: Complex/simple vocabulary
    - Type-token ratio: Unique words/total words
    - Semantic appropriateness: Analyze precision and context fit
    - Word frequency analysis: Identify most common words used

    B. Grammar and Morphology (systematic analysis):
    - Morphological complexity assessment
    - Derivational morpheme use: prefixes, suffixes
    - Inflectional morphology: plurals, tense, agreement
    - Compound word formation
    - Error pattern analysis by morpheme type

    3. COMPLEX SENTENCE ANALYSIS

    A. Sentence Structure Distribution:
    - Simple sentences ([SIMPLE_SENT]): Count and calculate percentage
      * Subject + predicate: "I went home"
      * Analyze average length and complexity
    - Complex sentences ([COMPLEX_SENT]): Count subordination patterns
      * Adverbial clauses: "When I got home, I ate dinner"
      * Relative clauses: "The bike that I rode was red"
      * Noun clauses: "I know that he likes pizza"
    - Compound sentences ([COMPOUND_SENT]): Count coordination patterns
      * Coordinating conjunctions: "and," "but," "or," "so"
      * Analyze balance and appropriateness

    B. Syntactic Complexity Measures:
    - Mean Length of Utterance (MLU): Words and morphemes
    - Clauses per utterance ratio
    - Subordination index
    - Coordination index

    4. FIGURATIVE LANGUAGE ANALYSIS

    A. Non-literal Language Use:
    - Figurative expressions ([FIGURATIVE]): Count and analyze
      * Metaphors: "Time is money"
      * Similes: "Fast as lightning"
      * Idioms: "Raining cats and dogs"
    - Appropriateness assessment: Context only
    - Comprehension vs. production abilities
    - Abstract language development indicators

    5. PRAGMATIC LANGUAGE ASSESSMENT

    A. Discourse Management:
    - Topic management ([TOPIC_SHIFT]): Count and assess appropriateness
      * Smooth transitions vs. abrupt shifts
      * Topic maintenance duration
      * Elaboration and detail provision
    - Tangential speech ([TANGENT]): Count off-topic instances
    - Discourse coherence ([COHERENCE_BREAK]): Analyze logical flow
    - Narrative structure and organization

    B. Referential Communication:
    - Referential clarity ([PRONOUN_REF]): Count unclear references
      * Ambiguous pronouns: "He told him that he was wrong"
      * Missing referents: "It was really good" (unclear antecedent)
    - Demonstrative use: "this," "that," "these," "those"
    - Overall conversational competence assessment

    6. VOCABULARY AND SEMANTIC ANALYSIS

    A. Semantic Accuracy and Precision:
    - Semantic errors ([SEMANTIC_ERROR]): Count inappropriate word choices
      * Word substitutions: "I drove my bicycle"
      * Category errors: "I petted the bird" (for touched)
    - Word association patterns and semantic relationships
    - Semantic categories: Analyze breadth and organization
    - Precision of word choice: Specific vs. general terms

    B. Lexical Diversity and Sophistication:
    - Vocabulary breadth: Range of semantic categories
    - Vocabulary depth: Precision and nuance within categories
    - Academic vs. conversational vocabulary ratio
    - Vocabulary development patterns observed

    7. NLP-DERIVED LINGUISTIC FEATURES (use bullet lists, NO tables)

    A. Lexical Diversity Measures (provide exact calculations as bullet points):
    - Type-Token Ratio (TTR): Unique words divided by total words
      * Calculate: [unique words] / [total words] = [ratio]
      * Interpretation: Higher ratios indicate greater lexical diversity
    - Moving Average Type-Token Ratio (MATTR): Average TTR across text segments
      * Calculate and interpret stability of lexical diversity
    - Measure of Textual Lexical Diversity (MTLD): Length of text segments maintaining TTR threshold
      * Higher values indicate sustained lexical diversity
      * Provide exact MTLD score and interpretation
    - Hypergeometric Distribution D (HDD): Probability-based diversity measure
      * Controls for text length effects
      * Provide HDD score

    B. Word Frequency Analysis (as bullet list, not table):
    - Most frequent words used: List top 10 as "word (count)" format
    - High-frequency vs. low-frequency word distribution
    - Function words vs. content words ratio
    - Repetitive word patterns observed

    C. Linguistic Complexity Indicators (bullet format):
    - Average word length in syllables
    - Syllable complexity patterns
    - Morphological complexity index
    - Syntactic complexity derived from automated parsing

    8. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS

    A. Morphological Patterns:
    - Derivational morphology: Prefixes and suffixes
    - Inflectional morphology: Tense, number, case markers
    - Morphological awareness indicators
    - Error patterns observed

    B. Phonological Considerations:
    - Sound pattern analysis (if evident in transcript)
    - Syllable structure complexity
    - Phonological awareness indicators

    9. COGNITIVE-LINGUISTIC FACTORS

    A. Working Memory Indicators:
    - Sentence length and complexity management
    - Information retention across utterances
    - Complex information processing evidence

    B. Processing Speed and Efficiency:
    - Word-finding speed and accuracy
    - Response latency patterns
    - Processing load indicators

    C. Executive Function Evidence:
    - Self-monitoring and error correction
    - Planning and organization in discourse
    CRITICAL REQUIREMENTS:
    1. Complete ALL 8 sections - do not stop early
    2. Provide exact counts for all markers with specific examples
    3. Calculate all percentages and rates with formulas shown
    4. Include direct quotes from transcript for examples
    5. Focus on objective data only - NO clinical interpretations or age/gender comparisons
    6. NO treatment recommendations or clinical implications
    7. If response is incomplete, end with <CONTINUE>
    8. FORMATTING: Use NO asterisks (**), NO hashtags (###), NO bolding - plain text only
    """
    
    return call_claude_api_with_continuation(analysis_prompt)

def calculate_linguistic_metrics(transcript_text):
    """Calculate comprehensive linguistic metrics from transcript"""
    import re
    import numpy as np
    
    if not transcript_text or not transcript_text.strip():
        return {}
    
    # Clean text and extract words
    cleaned_text = re.sub(r'\[.*?\]', '', transcript_text)  # Remove annotation markers
    sentences = re.split(r'[.!?]+', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Extract all words
    all_words = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        all_words.extend(words)
    
    if not all_words:
        return {}
    
    # Basic counts
    total_words = len(all_words)
    total_sentences = len(sentences)
    unique_words = len(set(all_words))
    
    # Type-Token Ratio
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Mean Length of Utterance (MLU)
    mlu_words = total_words / total_sentences if total_sentences > 0 else 0
    
    # Word frequency analysis
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    
    # Sentence length statistics
    sentence_lengths = []
    for sentence in sentences:
        words_in_sentence = len(re.findall(r'\b\w+\b', sentence))
        sentence_lengths.append(words_in_sentence)
    
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    std_sentence_length = np.std(sentence_lengths) if sentence_lengths else 0
    
    # Vocabulary sophistication (words > 6 characters as proxy for complex vocabulary)
    complex_words = [word for word in all_words if len(word) > 6]
    vocabulary_sophistication = len(complex_words) / total_words if total_words > 0 else 0
    
    # Calculate morpheme count (approximate)
    morpheme_count = 0
    for word in all_words:
        # Basic morpheme counting (word + common suffixes)
        morpheme_count += 1
        if word.endswith(('s', 'ed', 'ing', 'er', 'est', 'ly')):
            morpheme_count += 1
        if word.endswith(('tion', 'sion', 'ness', 'ment', 'able', 'ible')):
            morpheme_count += 1
    
    mlu_morphemes = morpheme_count / total_sentences if total_sentences > 0 else 0
    
    return {
        'total_words': total_words,
        'total_sentences': total_sentences,
        'unique_words': unique_words,
        'type_token_ratio': round(ttr, 3),
        'mlu_words': round(mlu_words, 2),
        'mlu_morphemes': round(mlu_morphemes, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'sentence_length_std': round(std_sentence_length, 2),
        'vocabulary_sophistication': round(vocabulary_sophistication, 3),
        'word_frequency': dict(list(sorted_word_freq.items())[:20]),  # Top 20 most frequent
        'sentence_lengths': sentence_lengths,
        'complex_word_count': len(complex_words),
        'morpheme_count': morpheme_count,
        'tokenized_words': all_words,  # Add for lexical diversity analysis
        'cleaned_text': cleaned_text   # Add for lexical diversity analysis
    }

def calculate_advanced_lexical_diversity(transcript_text):
    """Calculate advanced lexical diversity measures using lexical-diversity library"""
    import re
    
    try:
        from lexical_diversity import lex_div as ld
        lexdiv_available = True
    except ImportError:
        lexdiv_available = False
        
    if not lexdiv_available:
        return {
            'library_available': False,
            'error': 'lexical-diversity library not installed. Install with: pip install lexical-diversity'
        }
    
    if not transcript_text or not transcript_text.strip():
        return {'library_available': True, 'error': 'No text provided'}
    
    # Clean text and prepare for lexical diversity analysis
    cleaned_text = re.sub(r'\[.*?\]', '', transcript_text)  # Remove annotation markers
    
    try:
        # Tokenize using lexical-diversity
        tokens = ld.tokenize(cleaned_text)
        
        if len(tokens) < 10:  # Need minimum tokens for meaningful analysis
            return {
                'library_available': True,
                'error': f'Insufficient tokens for analysis (need ≥10, got {len(tokens)})'
            }
        
        # Calculate various lexical diversity measures
        diversity_measures = {}
        
        # Basic TTR (included for comparison, but noted as unreliable)
        diversity_measures['simple_ttr'] = round(ld.ttr(tokens), 4)
        
        # Recommended measures
        try:
            diversity_measures['root_ttr'] = round(ld.root_ttr(tokens), 4)
        except:
            diversity_measures['root_ttr'] = None
            
        try:
            diversity_measures['log_ttr'] = round(ld.log_ttr(tokens), 4)
        except:
            diversity_measures['log_ttr'] = None
            
        try:
            diversity_measures['maas_ttr'] = round(ld.maas_ttr(tokens), 4)
        except:
            diversity_measures['maas_ttr'] = None
        
        # MSTTR (Mean Segmental TTR) - only if enough tokens
        if len(tokens) >= 50:
            try:
                diversity_measures['msttr_50'] = round(ld.msttr(tokens, window_length=50), 4)
            except:
                diversity_measures['msttr_50'] = None
        
        if len(tokens) >= 25:
            try:
                diversity_measures['msttr_25'] = round(ld.msttr(tokens, window_length=25), 4)
            except:
                diversity_measures['msttr_25'] = None
        
        # MATTR (Moving Average TTR) - only if enough tokens
        if len(tokens) >= 50:
            try:
                diversity_measures['mattr_50'] = round(ld.mattr(tokens, window_length=50), 4)
            except:
                diversity_measures['mattr_50'] = None
                
        if len(tokens) >= 25:
            try:
                diversity_measures['mattr_25'] = round(ld.mattr(tokens, window_length=25), 4)
            except:
                diversity_measures['mattr_25'] = None
        
        # HDD (Hypergeometric Distribution D)
        try:
            diversity_measures['hdd'] = round(ld.hdd(tokens), 4)
        except:
            diversity_measures['hdd'] = None
        
        # MTLD (Measure of Textual Lexical Diversity) - only if enough tokens
        if len(tokens) >= 50:
            try:
                diversity_measures['mtld'] = round(ld.mtld(tokens), 4)
            except:
                diversity_measures['mtld'] = None
                
            try:
                diversity_measures['mtld_ma_wrap'] = round(ld.mtld_ma_wrap(tokens), 4)
            except:
                diversity_measures['mtld_ma_wrap'] = None
                
            try:
                diversity_measures['mtld_ma_bid'] = round(ld.mtld_ma_bid(tokens), 4)
            except:
                diversity_measures['mtld_ma_bid'] = None
        
        return {
            'library_available': True,
            'token_count': len(tokens),
            'diversity_measures': diversity_measures,
            'tokens': tokens[:50]  # First 50 tokens for verification
        }
        
    except Exception as e:
        return {
            'library_available': True,
            'error': f'Error calculating lexical diversity: {str(e)}'
        }

def analyze_annotation_markers(annotated_transcript):
    """Analyze and count all annotation markers in the transcript with detailed word-level analysis"""
    import re
    
    if not annotated_transcript:
        return {}
    
    # Define all marker types
    marker_types = {
        'FILLER': r'\[FILLER\]',
        'FALSE_START': r'\[FALSE_START\]',
        'REPETITION': r'\[REPETITION\]',
        'REVISION': r'\[REVISION\]',
        'PAUSE': r'\[PAUSE\]',
        'CIRCUMLOCUTION': r'\[CIRCUMLOCUTION\]',
        'INCOMPLETE': r'\[INCOMPLETE\]',
        'GENERIC': r'\[GENERIC\]',
        'WORD_SEARCH': r'\[WORD_SEARCH\]',
        'GRAM_ERROR': r'\[GRAM_ERROR\]',
        'SYNTAX_ERROR': r'\[SYNTAX_ERROR\]',
        'MORPH_ERROR': r'\[MORPH_ERROR\]',
        'RUN_ON': r'\[RUN_ON\]',
        'SIMPLE_VOCAB': r'\[SIMPLE_VOCAB\]',
        'COMPLEX_VOCAB': r'\[COMPLEX_VOCAB\]',
        'SEMANTIC_ERROR': r'\[SEMANTIC_ERROR\]',
        'TOPIC_SHIFT': r'\[TOPIC_SHIFT\]',
        'TANGENT': r'\[TANGENT\]',
        'INAPPROPRIATE': r'\[INAPPROPRIATE\]',
        'COHERENCE_BREAK': r'\[COHERENCE_BREAK\]',
        'SIMPLE_SENT': r'\[SIMPLE_SENT\]',
        'COMPLEX_SENT': r'\[COMPLEX_SENT\]',
        'COMPOUND_SENT': r'\[COMPOUND_SENT\]',
        'FIGURATIVE': r'\[FIGURATIVE\]',
        'PRONOUN_REF': r'\[PRONOUN_REF\]',
        'MAZING': r'\[MAZING\]',
        'PERSEVERATION': r'\[PERSEVERATION\]'
    }
    
    # Count each marker type and extract the actual words
    marker_counts = {}
    marker_examples = {}
    marker_words = {}
    
    for marker_name, pattern in marker_types.items():
        matches = re.findall(pattern, annotated_transcript)
        marker_counts[marker_name] = len(matches)
        
        # Find examples with context and extract the actual words
        examples = []
        words = []
        
        # Find all instances of word[MARKER] pattern
        word_pattern = r'(\w+)' + pattern
        word_matches = re.finditer(word_pattern, annotated_transcript)
        
        for match in word_matches:
            word = match.group(1)
            words.append(word)
            
            # Get context around the match
            start = max(0, match.start() - 30)
            end = min(len(annotated_transcript), match.end() + 30)
            context = annotated_transcript[start:end].strip()
            examples.append(f'"{word}" in context: {context}')
        
        marker_examples[marker_name] = examples[:10]  # Keep first 10 examples
        marker_words[marker_name] = words
    
    # Calculate totals by category
    fluency_total = sum([marker_counts.get(m, 0) for m in ['FILLER', 'FALSE_START', 'REPETITION', 'REVISION', 'PAUSE']])
    grammar_total = sum([marker_counts.get(m, 0) for m in ['GRAM_ERROR', 'SYNTAX_ERROR', 'MORPH_ERROR', 'RUN_ON']])
    vocab_simple = marker_counts.get('SIMPLE_VOCAB', 0)
    vocab_complex = marker_counts.get('COMPLEX_VOCAB', 0)
    
    return {
        'marker_counts': marker_counts,
        'marker_examples': marker_examples,
        'marker_words': marker_words,
        'category_totals': {
            'fluency_issues': fluency_total,
            'grammar_errors': grammar_total,
            'simple_vocabulary': vocab_simple,
            'complex_vocabulary': vocab_complex,
            'vocab_sophistication_ratio': vocab_complex / (vocab_simple + vocab_complex) if (vocab_simple + vocab_complex) > 0 else 0
        }
    }

def generate_comprehensive_analysis_report(annotated_transcript, original_transcript):
    """Generate the most comprehensive analysis combining manual counts, lexical diversity, and linguistic metrics"""
    import re
    
    if not annotated_transcript:
        return "No annotated transcript provided."
    
    # Get all three types of analysis
    linguistic_metrics = calculate_linguistic_metrics(original_transcript)
    marker_analysis = analyze_annotation_markers(annotated_transcript)
    lexical_diversity = calculate_advanced_lexical_diversity(original_transcript)
    
    # Calculate rates per 100 words
    total_words = linguistic_metrics.get('total_words', 0)
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("COMPREHENSIVE SPEECH ANALYSIS REPORT")
    report_lines.append("Combining Manual Counts + Advanced Lexical Diversity + Linguistic Metrics")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    # SECTION 1: BASIC STATISTICS
    report_lines.append("1. BASIC STATISTICS:")
    report_lines.append(f"   • Total words: {total_words}")
    report_lines.append(f"   • Total sentences: {linguistic_metrics.get('total_sentences', 0)}")
    report_lines.append(f"   • Unique words: {linguistic_metrics.get('unique_words', 0)}")
    report_lines.append(f"   • MLU (words): {linguistic_metrics.get('mlu_words', 0):.2f}")
    report_lines.append(f"   • MLU (morphemes): {linguistic_metrics.get('mlu_morphemes', 0):.2f}")
    report_lines.append(f"   • Average sentence length: {linguistic_metrics.get('avg_sentence_length', 0):.2f}")
    report_lines.append("")
    
    # SECTION 2: ADVANCED LEXICAL DIVERSITY MEASURES
    report_lines.append("2. ADVANCED LEXICAL DIVERSITY MEASURES:")
    if lexical_diversity.get('library_available', False) and 'diversity_measures' in lexical_diversity:
        measures = lexical_diversity['diversity_measures']
        report_lines.append(f"   • Token count for analysis: {lexical_diversity.get('token_count', 0)}")
        report_lines.append("")
        report_lines.append("   RECOMMENDED MEASURES:")
        
        if measures.get('root_ttr') is not None:
            report_lines.append(f"   • Root TTR: {measures['root_ttr']:.4f}")
        if measures.get('log_ttr') is not None:
            report_lines.append(f"   • Log TTR: {measures['log_ttr']:.4f}")
        if measures.get('maas_ttr') is not None:
            report_lines.append(f"   • Maas TTR: {measures['maas_ttr']:.4f}")
        if measures.get('hdd') is not None:
            report_lines.append(f"   • HDD (Hypergeometric Distribution D): {measures['hdd']:.4f}")
        
        report_lines.append("")
        report_lines.append("   MOVING WINDOW MEASURES:")
        if measures.get('msttr_25') is not None:
            report_lines.append(f"   • MSTTR (25-word window): {measures['msttr_25']:.4f}")
        if measures.get('msttr_50') is not None:
            report_lines.append(f"   • MSTTR (50-word window): {measures['msttr_50']:.4f}")
        if measures.get('mattr_25') is not None:
            report_lines.append(f"   • MATTR (25-word window): {measures['mattr_25']:.4f}")
        if measures.get('mattr_50') is not None:
            report_lines.append(f"   • MATTR (50-word window): {measures['mattr_50']:.4f}")
        
        report_lines.append("")
        report_lines.append("   MTLD MEASURES:")
        if measures.get('mtld') is not None:
            report_lines.append(f"   • MTLD: {measures['mtld']:.4f}")
        if measures.get('mtld_ma_wrap') is not None:
            report_lines.append(f"   • MTLD (moving average, wrap): {measures['mtld_ma_wrap']:.4f}")
        if measures.get('mtld_ma_bid') is not None:
            report_lines.append(f"   • MTLD (moving average, bidirectional): {measures['mtld_ma_bid']:.4f}")
        
        report_lines.append("")
        report_lines.append("   COMPARISON MEASURE:")
        report_lines.append(f"   • Simple TTR (not recommended): {measures.get('simple_ttr', 0):.4f}")
        
    else:
        report_lines.append("   Advanced lexical diversity measures not available")
        if 'error' in lexical_diversity:
            report_lines.append(f"   Error: {lexical_diversity['error']}")
    
    report_lines.append("")
    
    # SECTION 3: MANUAL ANNOTATION COUNTS
    report_lines.append("3. MANUAL ANNOTATION COUNTS:")
    marker_counts = marker_analysis['marker_counts']
    marker_words = marker_analysis['marker_words']
    
    # Group markers by category for organized reporting
    categories = {
        'FLUENCY MARKERS': ['FILLER', 'FALSE_START', 'REPETITION', 'REVISION', 'PAUSE'],
        'WORD RETRIEVAL MARKERS': ['CIRCUMLOCUTION', 'INCOMPLETE', 'GENERIC', 'WORD_SEARCH'],
        'GRAMMAR MARKERS': ['GRAM_ERROR', 'SYNTAX_ERROR', 'MORPH_ERROR', 'RUN_ON'],
        'VOCABULARY MARKERS': ['SIMPLE_VOCAB', 'COMPLEX_VOCAB', 'SEMANTIC_ERROR'],
        'PRAGMATIC MARKERS': ['TOPIC_SHIFT', 'TANGENT', 'INAPPROPRIATE', 'COHERENCE_BREAK', 'PRONOUN_REF'],
        'SENTENCE COMPLEXITY MARKERS': ['SIMPLE_SENT', 'COMPLEX_SENT', 'COMPOUND_SENT', 'FIGURATIVE'],
        'OTHER MARKERS': ['MAZING', 'PERSEVERATION']
    }
    
    for category, markers in categories.items():
        category_total = sum(marker_counts.get(marker, 0) for marker in markers)
        if category_total > 0:
            report_lines.append(f"   {category}:")
            
            for marker in markers:
                count = marker_counts.get(marker, 0)
                if count > 0:
                    rate = (count / total_words * 100) if total_words > 0 else 0
                    words_list = marker_words.get(marker, [])
                    
                    report_lines.append(f"     • {marker}: {count} instances ({rate:.2f} per 100 words)")
                    
                    if words_list:
                        # Count frequency of each word
                        word_freq = {}
                        for word in words_list:
                            word_freq[word] = word_freq.get(word, 0) + 1
                        
                        # Sort by frequency
                        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                        word_summary = []
                        for word, freq in sorted_words[:8]:  # Top 8 most frequent
                            if freq > 1:
                                word_summary.append(f'"{word}" ({freq}x)')
                            else:
                                word_summary.append(f'"{word}"')
                        
                        report_lines.append(f"       Words: {', '.join(word_summary)}")
            
            report_lines.append(f"     CATEGORY TOTAL: {category_total} instances")
            report_lines.append("")
    
    # SECTION 4: SUMMARY STATISTICS
    report_lines.append("4. SUMMARY STATISTICS:")
    category_totals = marker_analysis['category_totals']
    
    fluency_total = category_totals['fluency_issues']
    grammar_total = category_totals['grammar_errors']
    simple_vocab = category_totals['simple_vocabulary']
    complex_vocab = category_totals['complex_vocabulary']
    
    if total_words > 0:
        report_lines.append(f"   • Total fluency issues: {fluency_total} ({fluency_total/total_words*100:.2f} per 100 words)")
        report_lines.append(f"   • Total grammar errors: {grammar_total} ({grammar_total/total_words*100:.2f} per 100 words)")
        report_lines.append(f"   • Simple vocabulary: {simple_vocab} ({simple_vocab/total_words*100:.2f} per 100 words)")
        report_lines.append(f"   • Complex vocabulary: {complex_vocab} ({complex_vocab/total_words*100:.2f} per 100 words)")
        
        if simple_vocab + complex_vocab > 0:
            vocab_ratio = complex_vocab / (simple_vocab + complex_vocab)
            report_lines.append(f"   • Vocabulary sophistication ratio: {vocab_ratio:.3f}")
    
    # SECTION 5: WORD FREQUENCY ANALYSIS
    word_freq = linguistic_metrics.get('word_frequency', {})
    if word_freq:
        report_lines.append("")
        report_lines.append("5. MOST FREQUENT WORDS:")
        for i, (word, freq) in enumerate(list(word_freq.items())[:15], 1):
            percentage = (freq / total_words * 100) if total_words > 0 else 0
            report_lines.append(f"   {i:2d}. '{word}': {freq} times ({percentage:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("END OF COMPREHENSIVE ANALYSIS REPORT")
    report_lines.append("=" * 100)
    
    return '\n'.join(report_lines)

def generate_manual_count_report(annotated_transcript):
    """Generate a basic manual count report (legacy function for compatibility)"""
    return generate_comprehensive_analysis_report(annotated_transcript, annotated_transcript)

def process_file(file):
    """Process uploaded transcript file"""
    if file is None:
        return "Please upload a file first."
    
    try:
        with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return "File appears to be empty."
            
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def call_claude_api_with_continuation(prompt):
    """Call Claude API with smart continuation system - unlimited continuations until complete"""
    if not ANTHROPIC_API_KEY:
        return "Error: Claude API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    print("Starting comprehensive 13-section analysis...")
    print("This may take 3-5 minutes for complex analyses...")
    
    # Define all required sections
    required_sections = [
        "1. SPEECH FACTORS",
        "2. LANGUAGE SKILLS ASSESSMENT", 
        "3. COMPLEX SENTENCE ANALYSIS",
        "4. FIGURATIVE LANGUAGE ANALYSIS",
        "5. PRAGMATIC LANGUAGE ASSESSMENT",
        "6. VOCABULARY AND SEMANTIC ANALYSIS",
        "7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS",
        "8. QUANTITATIVE METRICS AND NLP FEATURES"
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
        initial_prompt = prompt + "\n\nCRITICAL INSTRUCTIONS: You MUST complete ALL 13 sections of the analysis. If your response is cut off or incomplete, end with <CONTINUE> to indicate more content is needed. Do not skip any sections. Use the checklist to ensure all sections are completed."
        
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
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": current_prompt
                    }
                ]
            }
            
            # Retry logic for timeout errors
            max_retries = 3
            retry_count = 0
            response = None
            
            while retry_count < max_retries:
                try:
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=data,
                        timeout=180
                    )
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Timeout occurred, retrying ({retry_count}/{max_retries})...")
                        time.sleep(5)  # Wait 5 seconds before retry
                    else:
                        print(f"Max retries ({max_retries}) exceeded due to timeouts")
                        return f"Error: API timeout after {max_retries} attempts. The analysis request is too complex. Try using 'Targeted Analysis' for specific sections."
                except Exception as e:
                    print(f"API call failed: {str(e)}")
                    return f"Error: {str(e)}"
            
            if response and response.status_code == 200:
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
                has_continue_marker = "<CONTINUE>" in response_text
                has_missing_sections = len(missing_sections) > 0
                
                # Continuation needed if either marker present OR sections missing
                needs_continuation = has_continue_marker or has_missing_sections
                
                print(f"Has <CONTINUE> marker: {has_continue_marker}")
                print(f"Has missing sections: {has_missing_sections}")
                print(f"Missing sections: {missing_sections}")
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
                return f"Error: Claude API Error: {response.status_code}"
                
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return f"Error: {str(e)}"
    
    # Combine all sections in the correct order
    final_response = combine_sections_smartly(all_sections)
    
    # Clean formatting: remove asterisks, hashtags, and fix table formatting
    final_response = clean_output_formatting(final_response)
    
    # Log final results
    print(f"\n=== FINAL SMART VALIDATION ===")
    print(f"Total sections found: {len(all_sections)}")
    print(f"All sections present: {len(all_sections) == 13}")
    print(f"Missing sections: {[s for s in required_sections if s not in all_sections]}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Total API calls: {continuation_count + 1}")
    print("=" * 50)
    
    # Add completion message
    if len(all_sections) == 13:
        print("ANALYSIS COMPLETE - All 13 sections generated successfully!")
        print("Output has been cleaned (removed asterisks, hashtags, converted tables to lists)")
    else:
        print(f"ANALYSIS INCOMPLETE - {13 - len(all_sections)} sections missing")
    
    # Add completion indicator with safety info
    if continuation_count > 0:
        final_response += f"\n\n[Analysis completed in {continuation_count + 1} parts over {(time.time() - start_time) / 60:.1f} minutes]"
    
    # Add warning if incomplete due to safety limits
    if len(all_sections) < 13:
        missing_sections = [s for s in required_sections if s not in all_sections]
        final_response += f"\n\nWARNING: Analysis incomplete due to safety limits. Missing sections: {', '.join(missing_sections)}"
        final_response += f"\n\nTIP: Try running the analysis again, or use the 'Targeted Analysis' tab to focus on specific areas."
        final_response += f"\nThe 'Quick Questions' tab may also provide faster results for specific areas of interest."
    
    return final_response

def analyze_with_backup(annotated_transcript, original_transcript, age, gender, slp_notes):
    """Analyze annotated transcript with original as backup"""
    if not annotated_transcript or len(annotated_transcript.strip()) < 50:
        return "Error: Please provide an annotated transcript for analysis."
    
    # Add SLP notes to the prompt if provided
    notes_section = ""
    if slp_notes and slp_notes.strip():
        notes_section = f"""
    
    SLP CLINICAL NOTES:
    {slp_notes.strip()}
    """
    
    # Calculate quantitative metrics
    linguistic_metrics = calculate_linguistic_metrics(original_transcript)
    marker_analysis = analyze_annotation_markers(annotated_transcript)
    
    # Format metrics for inclusion in prompt
    metrics_text = f"""
    
    CALCULATED LINGUISTIC METRICS:
    - Total Words: {linguistic_metrics.get('total_words', 0)}
    - Total Sentences: {linguistic_metrics.get('total_sentences', 0)}
    - Unique Words: {linguistic_metrics.get('unique_words', 0)}
    - Type-Token Ratio: {linguistic_metrics.get('type_token_ratio', 0)}
    - MLU (Words): {linguistic_metrics.get('mlu_words', 0)}
    - MLU (Morphemes): {linguistic_metrics.get('mlu_morphemes', 0)}
    - Average Sentence Length: {linguistic_metrics.get('avg_sentence_length', 0)}
    - Vocabulary Sophistication: {linguistic_metrics.get('vocabulary_sophistication', 0)}
    
    ANNOTATION MARKER COUNTS:
    - Fluency Issues: {marker_analysis.get('category_totals', {}).get('fluency_issues', 0)}
    - Grammar Errors: {marker_analysis.get('category_totals', {}).get('grammar_errors', 0)}
    - Simple Vocabulary: {marker_analysis.get('category_totals', {}).get('simple_vocabulary', 0)}
    - Complex Vocabulary: {marker_analysis.get('category_totals', {}).get('complex_vocabulary', 0)}
    - Vocabulary Sophistication Ratio: {marker_analysis.get('category_totals', {}).get('vocab_sophistication_ratio', 0):.3f}
    """
    
    analysis_prompt = f"""
    You are a speech-language pathologist conducting a COMPREHENSIVE analysis of a word-by-word annotated speech sample. Use the provided quantitative metrics and count EVERY marker precisely.

    Patient: {age}-year-old {gender}
    
    ANNOTATED TRANSCRIPT:
    {annotated_transcript}{notes_section}
    
    ORIGINAL TRANSCRIPT (for reference and backup analysis):
    {original_transcript}
    
    {metrics_text}
    
    ANALYSIS INSTRUCTIONS:
    Using the detailed linguistic markers in the annotated transcript and the calculated metrics above, provide a comprehensive analysis with EXACT counts, percentages, and specific examples. Complete ALL 13 sections below:

    COMPREHENSIVE SPEECH SAMPLE ANALYSIS:

    1. SPEECH FACTORS (with EXACT counts and specific citations):

    A. Fluency Issues:
    - Count [FILLER] markers: List each instance and calculate rate per 100 words
    - Count [FALSE_START] markers: List examples and analyze patterns
    - Count [REPETITION] markers: Categorize by type (word, phrase, sound)
    - Count [REVISION] markers: Analyze self-correction patterns
    - Count [PAUSE] markers: Assess hesitation frequency
    - Calculate total disfluency rate

    B. Word Retrieval Issues:
    - Count [CIRCUMLOCUTION] markers: List each roundabout description
    - Count [INCOMPLETE] markers: Analyze abandoned thought patterns
    - Count [GENERIC] markers: Calculate specificity ratio
    - Count [WORD_SEARCH] markers: Identify retrieval difficulty areas

    C. Grammatical Errors:
    - Count [GRAM_ERROR] markers by subcategory (verb tense, subject-verb agreement, etc.)
    - Count [SYNTAX_ERROR] markers: Analyze word order problems
    - Count [MORPH_ERROR] markers: Categorize morphological mistakes
    - Count [RUN_ON] markers: Assess sentence boundary awareness

    2. LANGUAGE SKILLS ASSESSMENT (with specific evidence):

    A. Lexical/Semantic Skills:
    - Use calculated Type-Token Ratio: {linguistic_metrics.get('type_token_ratio', 0)}
    - Count [SIMPLE_VOCAB] vs [COMPLEX_VOCAB] markers
    - Assess vocabulary sophistication ratio: {marker_analysis.get('category_totals', {}).get('vocab_sophistication_ratio', 0):.3f}
    - Count [SEMANTIC_ERROR] markers and analyze patterns

    B. Syntactic Skills:
    - Count [SIMPLE_SENT], [COMPLEX_SENT], [COMPOUND_SENT] markers
    - Calculate sentence complexity ratios
    - Assess clause complexity and embedding

    C. Supralinguistic Skills:
    - Identify cause-effect relationships, inferences, non-literal language
    - Assess problem-solving language and metalinguistic awareness

    3. COMPLEX SENTENCE ANALYSIS (with exact counts):

    A. Coordinating Conjunctions:
    - Count and cite EVERY use of: and, but, or, so, yet, for, nor
    - Analyze patterns and age-appropriateness

    B. Subordinating Conjunctions:
    - Count and cite EVERY use of: because, although, while, since, if, when, where, that, which, who
    - Analyze clause complexity and embedding depth

    C. Sentence Structure Analysis:
    - Use calculated MLU: {linguistic_metrics.get('mlu_words', 0)} words, {linguistic_metrics.get('mlu_morphemes', 0)} morphemes
    - Calculate complexity ratios

    4. FIGURATIVE LANGUAGE ANALYSIS (with exact counts):

    A. Similes and Metaphors:
    - Count [FIGURATIVE] markers for similes (using "like" or "as")
    - Count [FIGURATIVE] markers for metaphors (direct comparisons)

    B. Idioms and Non-literal Language:
    - Count and analyze idiomatic expressions
    - Assess comprehension and appropriate use

    5. PRAGMATIC LANGUAGE ASSESSMENT (with specific examples):

    A. Discourse Management:
    - Count [TOPIC_SHIFT] markers: Assess transition appropriateness
    - Count [TANGENT] markers: Analyze tangential speech patterns
    - Count [COHERENCE_BREAK] markers: Assess logical flow

    B. Referential Communication:
    - Count [PRONOUN_REF] markers: Analyze referential clarity
    - Assess communicative effectiveness

    6. VOCABULARY AND SEMANTIC ANALYSIS (with quantification):

    A. Vocabulary Diversity:
    - Total words: {linguistic_metrics.get('total_words', 0)}
    - Unique words: {linguistic_metrics.get('unique_words', 0)}
    - Type-Token Ratio: {linguistic_metrics.get('type_token_ratio', 0)}
    - Vocabulary sophistication: {linguistic_metrics.get('vocabulary_sophistication', 0)}

    B. Semantic Relationships:
    - Analyze word frequency patterns
    - Assess semantic precision and relationships

    7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS (with counts):

    A. Morphological Markers:
    - Count [MORPH_ERROR] markers and categorize
    - Analyze morpheme use patterns
    - Assess morphological complexity

    B. Phonological Patterns:
    - Identify speech sound patterns from transcript
    - Assess syllable structure complexity

    8. COGNITIVE-LINGUISTIC FACTORS (with evidence):

    A. Working Memory:
    - Assess sentence length complexity using average: {linguistic_metrics.get('avg_sentence_length', 0)} words
    - Analyze information retention patterns

    B. Processing Efficiency:
    - Analyze linguistic complexity and word-finding patterns
    - Assess cognitive demands of language structures

    C. Executive Function:
    - Count self-correction patterns ([REVISION] markers)
    - Assess planning and organization in discourse

    9. FLUENCY AND RHYTHM ANALYSIS (with quantification):

    A. Disfluency Patterns:
    - Total fluency issues: {marker_analysis.get('category_totals', {}).get('fluency_issues', 0)}
    - Calculate disfluency rate per 100 words
    - Analyze impact on communication

    B. Language Flow:
    - Assess sentence length variability: std = {linguistic_metrics.get('sentence_length_std', 0)}
    - Analyze linguistic markers of hesitation

    10. QUANTITATIVE METRICS:
    - Total words: {linguistic_metrics.get('total_words', 0)}
    - Total sentences: {linguistic_metrics.get('total_sentences', 0)}
    - MLU (words): {linguistic_metrics.get('mlu_words', 0)}
    - MLU (morphemes): {linguistic_metrics.get('mlu_morphemes', 0)}
    - Type-Token Ratio: {linguistic_metrics.get('type_token_ratio', 0)}
    - Grammar error rate: Calculate from marker counts
    - Vocabulary sophistication ratio: {marker_analysis.get('category_totals', {}).get('vocab_sophistication_ratio', 0):.3f}

    CRITICAL REQUIREMENTS:
    - Use the provided calculated metrics in your analysis
    - Provide EXACT counts for every marker type
    - Calculate precise percentages and show your work
    - Give specific examples from the transcript
    - If annotation is incomplete, supplement with analysis of the original transcript
    - Complete ALL 8 sections - use <CONTINUE> if needed
    - Focus on objective data only - NO clinical interpretations
    """
    
    return call_claude_api_with_continuation(analysis_prompt)

def full_analysis_pipeline(transcript_content, age, gender, slp_notes, progress_callback=None):
    """Complete pipeline: annotate then analyze with progressive updates"""
    if not transcript_content or len(transcript_content.strip()) < 50:
        return "Error: Please provide a longer transcript for analysis.", ""
    
    # Step 1: Annotate transcript
    logger.info("Step 1: Annotating transcript with linguistic markers...")
    if progress_callback:
        progress_callback("Step 1: Annotating transcript with linguistic markers...")
    
    annotated_transcript = annotate_transcript(transcript_content, age, gender, slp_notes)
    
    if annotated_transcript.startswith("Error"):
        return annotated_transcript, ""
    
    # Return annotated transcript immediately
    if progress_callback:
        progress_callback("Step 1 Complete: Annotation finished! Starting analysis...")
    
    # Check if annotation was incomplete
    if annotated_transcript.startswith("ANNOTATION INCOMPLETE"):
        logger.warning("Annotation incomplete, proceeding with analysis using original transcript as primary source")
        analysis_note = "Note: Annotation was incomplete. Analysis primarily based on original transcript.\n\n"
    else:
        analysis_note = ""
    
    # Step 2: Analyze annotated transcript with original as backup
    logger.info("Step 2: Analyzing annotated transcript...")
    if progress_callback:
        progress_callback("Step 2: Analyzing annotated transcript (this may take several minutes)...")
    
    analysis_result = analyze_with_backup(annotated_transcript, transcript_content, age, gender, slp_notes)
    
    if progress_callback:
        progress_callback("Analysis Complete!")
    
    return annotated_transcript, analysis_note + analysis_result

def progressive_analysis_pipeline(transcript_content, age, gender, slp_notes):
    """Generator function for progressive analysis updates"""
    if not transcript_content or len(transcript_content.strip()) < 50:
        yield "Error: Please provide a longer transcript for analysis.", "", "Error"
        return
    
    # Step 1: Annotate transcript
    logger.info("Step 1: Annotating transcript with linguistic markers...")
    yield "", "", "Step 1: Annotating transcript with linguistic markers..."
    
    annotated_transcript = annotate_transcript(transcript_content, age, gender, slp_notes)
    
    if annotated_transcript.startswith("Error"):
        yield annotated_transcript, "", "Annotation failed"
        return
    
    # Return annotated transcript immediately after completion
    yield annotated_transcript, "", "Step 1 Complete! Starting analysis..."
    
    # Check if annotation was incomplete
    if annotated_transcript.startswith("ANNOTATION INCOMPLETE"):
        logger.warning("Annotation incomplete, proceeding with analysis")
        analysis_note = "Note: Annotation was incomplete. Analysis primarily based on original transcript.\n\n"
        yield annotated_transcript, "", "Annotation incomplete, continuing with analysis..."
    else:
        analysis_note = ""
    
    # Step 2: Analyze annotated transcript
    logger.info("Step 2: Analyzing annotated transcript...")
    yield annotated_transcript, "", "Step 2: Analyzing annotated transcript (this may take several minutes)..."
    
    analysis_result = analyze_with_backup(annotated_transcript, transcript_content, age, gender, slp_notes)
    
    # Final result
    yield annotated_transcript, analysis_note + analysis_result, "Analysis Complete!"

# Example transcript data
example_transcript = """Well, um, I was thinking about, you know, the thing that happened yesterday. I was go- I mean I was going to the store and, uh, I seen this really big dog. Actually, it was more like a wolf or something. The dog, he was just standing there, and I thought to myself, "That's one magnificent creature." But then, um, I realized I forgot my wallet at home, so I had to turn around and go back. When I got home, my wife she says to me, "Where's the groceries?" And I'm like, "Well, honey, I had to come back because I forgot my thing." She wasn't too happy about that, let me tell you. Anyway, speaking of dogs, did I ever tell you about the time I went fishing? It was raining cats and dogs that day, and I caught three fishes. My brother, he don't like fishing much, but he came with me anyway. We was sitting there for hours, just waiting and waiting. The fish, they wasn't biting at all. But then, all of a sudden, I got a bite! I was so excited, I almost falled into the water. The fish was huge - well, maybe not huge, but pretty big for that lake. We cooked it up real good that night. My wife, she made some of that fancy stuff to go with it. What do you call it... that green thing... oh yeah, asparagus. She's always making these elaborate meals. Sometimes I think she tries too hard, you know? But I appreciate it. Life's been good to us, I guess. We been married for twenty-five years now. Time flies when you're having fun, as they say."""

example_annotated = """Well[FILLER], um[FILLER], I was thinking about, you[SIMPLE_VOCAB] know[FILLER], the thing[GENERIC] that happened yesterday[SIMPLE_VOCAB]. I was go-[FALSE_START] I mean I was going[SIMPLE_VOCAB] to the store[SIMPLE_VOCAB] and, uh[FILLER], I seen[GRAM_ERROR] this really big[SIMPLE_VOCAB] dog[SIMPLE_VOCAB].[SIMPLE_SENT] Actually, it was more like[FILLER] a wolf[SIMPLE_VOCAB] or something[GENERIC].[SIMPLE_SENT] The dog[SIMPLE_VOCAB], he[PRONOUN_REF] was just standing[SIMPLE_VOCAB] there, and I thought to myself, "That's one magnificent[COMPLEX_VOCAB] creature[COMPLEX_VOCAB]."[COMPLEX_SENT] But then, um[FILLER], I realized[COMPLEX_VOCAB] I forgot[SIMPLE_VOCAB] my wallet[SIMPLE_VOCAB] at home[SIMPLE_VOCAB], so I had to turn around and go[SIMPLE_VOCAB] back[SIMPLE_VOCAB].[COMPLEX_SENT] When I got home, my wife[SIMPLE_VOCAB] she[REPETITION] says[SIMPLE_VOCAB] to me, "Where's the groceries[SIMPLE_VOCAB]?"[COMPLEX_SENT] And I'm like[FILLER], "Well[FILLER], honey[SIMPLE_VOCAB], I had to come back because I forgot[SIMPLE_VOCAB] my thing[GENERIC]."[COMPLEX_SENT] She wasn't too happy[SIMPLE_VOCAB] about that, let me tell you.[SIMPLE_SENT] Anyway[TOPIC_SHIFT], speaking of dogs, did I ever tell you about the time I went fishing?[TANGENT][COMPLEX_SENT] It was raining cats and dogs[FIGURATIVE] that day, and I caught[SIMPLE_VOCAB] three fishes[MORPH_ERROR].[COMPOUND_SENT] My brother[SIMPLE_VOCAB], he[PRONOUN_REF] don't[GRAM_ERROR] like fishing[SIMPLE_VOCAB] much, but he came with me anyway[SIMPLE_VOCAB].[COMPLEX_SENT] We was[GRAM_ERROR] sitting[SIMPLE_VOCAB] there for hours[SIMPLE_VOCAB], just waiting[SIMPLE_VOCAB] and waiting[REPETITION].[SIMPLE_SENT] The fish[SIMPLE_VOCAB], they[PRONOUN_REF] wasn't[GRAM_ERROR] biting[SIMPLE_VOCAB] at all.[SIMPLE_SENT] But then, all of a sudden[SIMPLE_VOCAB], I got[SIMPLE_VOCAB] a bite[SIMPLE_VOCAB]![SIMPLE_SENT] I was so excited[SIMPLE_VOCAB], I almost falled[MORPH_ERROR] into the water[SIMPLE_VOCAB].[COMPLEX_SENT] The fish[SIMPLE_VOCAB] was huge[SIMPLE_VOCAB] - well[FILLER], maybe not huge[SIMPLE_VOCAB], but pretty big[SIMPLE_VOCAB] for that lake[SIMPLE_VOCAB].[REVISION][COMPLEX_SENT] We cooked[SIMPLE_VOCAB] it up real good[SIMPLE_VOCAB] that night[SIMPLE_VOCAB].[SIMPLE_SENT] My wife[SIMPLE_VOCAB], she[REPETITION] made some of that fancy[SIMPLE_VOCAB] stuff[GENERIC] to go[SIMPLE_VOCAB] with it.[SIMPLE_SENT] What do you call it... [WORD_SEARCH] that green[SIMPLE_VOCAB] thing[GENERIC]... [PAUSE] oh yeah, asparagus[COMPLEX_VOCAB].[CIRCUMLOCUTION] She's always making[SIMPLE_VOCAB] these elaborate[COMPLEX_VOCAB] meals[SIMPLE_VOCAB].[SIMPLE_SENT] Sometimes I think[SIMPLE_VOCAB] she tries[SIMPLE_VOCAB] too hard[SIMPLE_VOCAB], you know[FILLER]?[COMPLEX_SENT] But I appreciate[COMPLEX_VOCAB] it.[SIMPLE_SENT] Life's been good[SIMPLE_VOCAB] to us, I guess[SIMPLE_VOCAB].[SIMPLE_SENT] We been[GRAM_ERROR] married[SIMPLE_VOCAB] for twenty-five[COMPLEX_VOCAB] years[SIMPLE_VOCAB] now.[SIMPLE_SENT] Time flies when you're having fun[FIGURATIVE], as they say.[COMPLEX_SENT]"""

# Create Gradio interface
with gr.Blocks(title="Speech Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    #  Speech Analysis Tool with Annotations
    
    This tool performs a two-step comprehensive speech analysis:
    1. **Annotation**: Marks linguistic features in the transcript
    2. **Analysis**: Counts and analyzes the marked features for detailed assessment
    
    Upload a transcript or paste text below to begin the analysis.
    """)
    
    with gr.Tab("Full Analysis Pipeline"):
        gr.Markdown("### Complete two-step analysis: annotation followed by comprehensive analysis")
        
        with gr.Row():
            with gr.Column(scale=2):
                transcript_input = gr.Textbox(
                    label="Speech Transcript",
                    placeholder="Paste the speech transcript here...",
                    lines=10,
                    max_lines=20
                )
                
                file_input = gr.File(
                    label="Or upload transcript file",
                    file_types=[".txt", ".doc", ".docx"]
                )
                
                with gr.Row():
                    age_input = gr.Textbox(
                        label="Age",
                        placeholder="e.g., 45",
                        value="45"
                    )
                    gender_input = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Other"],
                        value="Male"
                    )
                
                slp_notes_input = gr.Textbox(
                    label="SLP Clinical Notes (Optional)",
                    placeholder="Add any relevant clinical observations...",
                    lines=3
                )
                
                example_btn = gr.Button("Load Example Transcript", variant="secondary", size="sm")
                
                # Single main analysis button
                ultimate_analysis_btn = gr.Button("Run Complete Speech Analysis", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                status_display = gr.Markdown("Ready to analyze transcript")
                
                annotated_output = gr.Textbox(
                    label="Step 1: Annotated Transcript (Complete = Yes, Incomplete = No)",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True
                )
                
                analysis_output = gr.Textbox(
                    label="Step 2: Comprehensive Analysis",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
    
    with gr.Tab("Annotation Only"):
        gr.Markdown("Step 1: Annotate transcript with linguistic markers")
        
        with gr.Row():
            with gr.Column():
                transcript_input_2 = gr.Textbox(
                    label="Speech Transcript",
                    placeholder="Paste the speech transcript here...",
                    lines=10
                )
                
                with gr.Row():
                    age_input_2 = gr.Textbox(label="Age", value="45")
                    gender_input_2 = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Other"],
                        value="Male"
                    )
                
                slp_notes_input_2 = gr.Textbox(
                    label="SLP Clinical Notes (Optional)",
                    lines=3
                )
                
                example_btn_2 = gr.Button("Load Example Transcript", variant="secondary", size="sm")
                annotate_btn = gr.Button("Annotate Transcript", variant="secondary")
            
            with gr.Column():
                annotation_output = gr.Textbox(
                    label="Annotated Transcript (Complete = Yes, Incomplete = No)",
                    lines=20,
                    show_copy_button=True
                )
    

    with gr.Tab("Quick Questions"):
        gr.Markdown("Ask specific questions about the transcript")
        
        with gr.Row():
            with gr.Column():
                transcript_input_4 = gr.Textbox(
                    label="Speech Transcript",
                    placeholder="Paste the speech transcript here...",
                    lines=8
                )
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How many filler words are used? What grammatical errors are present?",
                    lines=2
                )
                
                with gr.Row():
                    age_input_4 = gr.Textbox(label="Age", value="45")
                    gender_input_4 = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Other"],
                        value="Male"
                    )
                
                slp_notes_input_4 = gr.Textbox(
                    label="SLP Clinical Notes (Optional)",
                    lines=2
                )
                
                # Quick question examples
                gr.Markdown("Example Questions:")
                with gr.Row():
                    q1_btn = gr.Button("Count filler words", size="sm", variant="secondary")
                    q2_btn = gr.Button("Grammar errors?", size="sm", variant="secondary")
                    q3_btn = gr.Button("Vocabulary level?", size="sm", variant="secondary")
                
                with gr.Row():
                    q4_btn = gr.Button("Sentence complexity?", size="sm", variant="secondary")
                    q5_btn = gr.Button("Word finding issues?", size="sm", variant="secondary")
                    q6_btn = gr.Button("Fluency problems?", size="sm", variant="secondary")
                
                example_btn_4 = gr.Button("Load Example Transcript", variant="secondary", size="sm")
                ask_question_btn = gr.Button("Ask Question", variant="primary")
            
            with gr.Column():
                question_output = gr.Textbox(
                    label="Answer",
                    lines=15,
                    show_copy_button=True
                )
    
    with gr.Tab("Targeted Analysis"):
        gr.Markdown("Focus on specific areas of speech and language")
        
        with gr.Row():
            with gr.Column():
                transcript_input_5 = gr.Textbox(
                    label="Speech Transcript",
                    placeholder="Paste the speech transcript here...",
                    lines=8
                )
                
                analysis_area = gr.Dropdown(
                    label="Analysis Focus",
                    choices=[
                        "Fluency and Disfluencies",
                        "Grammar and Syntax", 
                        "Vocabulary and Semantics",
                        "Pragmatics and Discourse",
                        "Sentence Complexity",
                        "Word Finding and Retrieval"
                    ],
                    value="Fluency and Disfluencies"
                )
                
                with gr.Row():
                    age_input_5 = gr.Textbox(label="Age", value="45")
                    gender_input_5 = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Other"],
                        value="Male"
                    )
                
                slp_notes_input_5 = gr.Textbox(
                    label="SLP Clinical Notes (Optional)",
                    lines=2
                )
                
                example_btn_5 = gr.Button("Load Example Transcript", variant="secondary", size="sm")
                targeted_analysis_btn = gr.Button("Run Targeted Analysis", variant="primary")
            
            with gr.Column():
                targeted_output = gr.Textbox(
                    label="Targeted Analysis Results",
                    lines=15,
                    show_copy_button=True
                )
    
    # Event handlers - now all components are defined
    example_btn.click(fn=lambda: example_transcript, outputs=[transcript_input])
    example_btn_2.click(fn=lambda: example_transcript, outputs=[transcript_input_2])
    example_btn_4.click(fn=lambda: example_transcript, outputs=[transcript_input_4])
    example_btn_5.click(fn=lambda: example_transcript, outputs=[transcript_input_5])
    
    # Quick question button handlers
    q1_btn.click(fn=lambda: "How many filler words (um, uh, like, you know) are used in this transcript? Provide exact counts and examples.", outputs=[question_input])
    q2_btn.click(fn=lambda: "What grammatical errors are present in this transcript? List all errors with specific examples and corrections.", outputs=[question_input])
    q3_btn.click(fn=lambda: "What is the vocabulary level and sophistication in this transcript? Analyze word choice and complexity.", outputs=[question_input])
    q4_btn.click(fn=lambda: "How complex are the sentences in this transcript? Analyze sentence types and structures used.", outputs=[question_input])
    q5_btn.click(fn=lambda: "Are there any word-finding difficulties or retrieval issues? Identify specific examples and patterns.", outputs=[question_input])
    q6_btn.click(fn=lambda: "What fluency problems or disfluencies are present? Count and categorize all instances.", outputs=[question_input])
    
    file_input.change(
        fn=process_file,
        inputs=[file_input],
        outputs=[transcript_input]
    )
    
    def run_annotation_step(transcript_content, age, gender, slp_notes):
        """Run just the annotation step and return immediately"""
        if not transcript_content or len(transcript_content.strip()) < 50:
            return "Error: Please provide a longer transcript for annotation.", "Error"
        
        logger.info("Step 1: Annotating transcript with linguistic markers...")
        annotated_transcript = annotate_transcript(transcript_content, age, gender, slp_notes)
        
        if annotated_transcript.startswith("Error"):
            return annotated_transcript, "Annotation failed"
        elif annotated_transcript.startswith("ANNOTATION INCOMPLETE"):
            return annotated_transcript, "Annotation incomplete but proceeding"
        else:
            return annotated_transcript, "Annotation complete! Click 'Run Analysis' to continue."
    
    def run_analysis_step(annotated_transcript, original_transcript, age, gender, slp_notes):
        """Run the analysis step on the annotated transcript"""
        if not annotated_transcript or len(annotated_transcript.strip()) < 50:
            return "Error: Please provide an annotated transcript for analysis."
        
        logger.info("Step 2: Analyzing annotated transcript...")
        
        # Check if annotation was incomplete
        if annotated_transcript.startswith("ANNOTATION INCOMPLETE"):
            analysis_note = "Note: Annotation was incomplete. Analysis primarily based on original transcript.\n\n"
        else:
            analysis_note = ""
        
        analysis_result = analyze_with_backup(annotated_transcript, original_transcript, age, gender, slp_notes)
        return analysis_note + analysis_result
    
    def run_manual_count_only(annotated_transcript):
        """Generate only the manual count report without AI analysis"""
        if not annotated_transcript or len(annotated_transcript.strip()) < 50:
            return "Error: Please provide an annotated transcript for manual counting."
        
        return generate_manual_count_report(annotated_transcript)
    
    def run_verified_analysis(annotated_transcript, original_transcript, age, gender, slp_notes):
        """Run analysis with manual count verification"""
        if not annotated_transcript or len(annotated_transcript.strip()) < 50:
            return "Error: Please provide an annotated transcript for analysis."
        
        # Generate comprehensive analysis report first
        comprehensive_report = generate_comprehensive_analysis_report(annotated_transcript, original_transcript)
        
        # Get all the verified data
        marker_analysis = analyze_annotation_markers(annotated_transcript)
        linguistic_metrics = calculate_linguistic_metrics(original_transcript)
        lexical_diversity = calculate_advanced_lexical_diversity(original_transcript)
        
        # Create a comprehensive verified analysis prompt
        verified_prompt = f"""
        You are a speech-language pathologist conducting analysis based on COMPREHENSIVE VERIFIED DATA. 
        Do NOT recount anything - use ONLY the provided verified measurements below.

        Patient: {age}-year-old {gender}
        
        COMPREHENSIVE VERIFIED ANALYSIS DATA (DO NOT RECOUNT):
        {comprehensive_report}
        
        ANNOTATED TRANSCRIPT (for examples only, do not recount):
        {annotated_transcript}...
        
        INSTRUCTIONS: 
        Use ONLY the verified data provided above. Do NOT count or calculate anything yourself.
        
        Provide a comprehensive clinical interpretation organized into these sections:

        1. LEXICAL DIVERSITY DATA:
        - Report the advanced lexical diversity measures (MTLD, HDD, MATTR, etc.)
        - Provide objective data interpretation only

        2. FLUENCY PATTERN DATA:
        - Report fluency marker counts and rates
        - Provide objective data summary only

        3. GRAMMATICAL PATTERN DATA:
        - Report grammar error patterns from verified counts
        - Provide objective data summary only

        4. VOCABULARY AND SEMANTIC ANALYSIS:
        - Interpretation of vocabulary sophistication measures
        - Word frequency pattern analysis
        - Semantic precision assessment

        5. PRAGMATIC LANGUAGE EVALUATION:
        - Discourse coherence based on verified markers
        - Social communication effectiveness
        - Conversational competence

        6. OVERALL COMMUNICATION PROFILE:
        - Integration of all verified measures
        - Strengths and areas of need
        - Functional communication impact

        Focus on OBJECTIVE DATA INTERPRETATION only, not clinical significance. 
        All measurements are already verified and accurate.
        Cite specific examples from the transcript to support your observations.
        """
        
        ai_interpretation = call_claude_api(verified_prompt)
        
        return f"{comprehensive_report}\n\n{'='*100}\nCLINICAL INTERPRETATION BASED ON COMPREHENSIVE VERIFIED DATA\n{'='*100}\n\n{ai_interpretation}"
    
    def run_ultimate_analysis(annotated_transcript, original_transcript, age, gender, slp_notes):
        """Clean comprehensive analysis using verified statistical data"""
        if not annotated_transcript or len(annotated_transcript.strip()) < 50:
            return "Error: Please provide an annotated transcript for analysis."
        
        # Gather statistical data
        linguistic_metrics = calculate_linguistic_metrics(original_transcript)
        marker_analysis = analyze_annotation_markers(annotated_transcript)
        lexical_diversity = calculate_advanced_lexical_diversity(original_transcript)
        
        # Prepare verified statistics
        marker_counts = marker_analysis['marker_counts']
        category_totals = marker_analysis['category_totals']
        total_words = linguistic_metrics.get('total_words', 0)
        
        stats_summary = f"""
        VERIFIED STATISTICAL DATA:
        
        Basic Metrics:
        - Total words: {total_words}
        - Total sentences: {linguistic_metrics.get('total_sentences', 0)}
        - Unique words: {linguistic_metrics.get('unique_words', 0)}
        - MLU words: {linguistic_metrics.get('mlu_words', 0):.2f}
        - MLU morphemes: {linguistic_metrics.get('mlu_morphemes', 0):.2f}
        - Average sentence length: {linguistic_metrics.get('avg_sentence_length', 0):.2f}
        
        Annotation Counts:
        - Filler markers: {marker_counts.get('FILLER', 0)} ({marker_counts.get('FILLER', 0)/total_words*100:.2f} per 100 words)
        - False starts: {marker_counts.get('FALSE_START', 0)}
        - Repetitions: {marker_counts.get('REPETITION', 0)}
        - Grammar errors: {marker_counts.get('GRAM_ERROR', 0)}
        - Simple vocabulary: {marker_counts.get('SIMPLE_VOCAB', 0)}
        - Complex vocabulary: {marker_counts.get('COMPLEX_VOCAB', 0)}
        - Simple sentences: {marker_counts.get('SIMPLE_SENT', 0)}
        - Complex sentences: {marker_counts.get('COMPLEX_SENT', 0)}
        - Compound sentences: {marker_counts.get('COMPOUND_SENT', 0)}
        
        Category Totals:
        - Total fluency issues: {category_totals['fluency_issues']} ({category_totals['fluency_issues']/total_words*100:.2f} per 100 words)
        - Total grammar errors: {category_totals['grammar_errors']}
        - Vocabulary sophistication ratio: {category_totals['vocab_sophistication_ratio']:.3f}
        """
        
        if lexical_diversity.get('library_available', False) and 'diversity_measures' in lexical_diversity:
            measures = lexical_diversity['diversity_measures']
            stats_summary += f"""
        
        Lexical Diversity:
        - Simple TTR: {measures.get('simple_ttr', 'N/A')}
        - HDD: {measures.get('hdd', 'N/A')}
        - MTLD: {measures.get('mtld', 'N/A')}
        - MATTR: {measures.get('mattr_25', 'N/A')}
        """
        
        # Create comprehensive analysis prompt
        final_prompt = f"""
        You are a speech-language pathologist conducting a comprehensive speech analysis. Use the verified statistical data provided and complete ALL 13 sections with detailed structure.

        Patient: {age}-year-old {gender}
        
        {stats_summary}
        
        ANNOTATED TRANSCRIPT (for examples and quotes):
        {annotated_transcript}
        
        INSTRUCTIONS:
        1. Use ONLY the verified statistical values above - do not recount anything
        2. Complete ALL 13 sections without stopping
        3. Provide specific examples and quotes from the transcript
        4. Calculate rates and percentages using verified counts
        5. Focus on clinical interpretation and actionable insights
        6. If response is incomplete, end with <CONTINUE>
        
        COMPREHENSIVE SPEECH SAMPLE ANALYSIS

        1. SPEECH FACTORS

        A. Fluency Issues (use verified counts):
        - Filler words: Use verified count of {marker_counts.get('FILLER', 0)} fillers
          * Calculate rate per 100 words: {marker_counts.get('FILLER', 0)/total_words*100:.2f}%
          * Identify types and provide examples from transcript
          * Provide objective count summary
        - False starts: Use verified count of {marker_counts.get('FALSE_START', 0)}
          * Provide specific examples from transcript
          * Analyze patterns and self-correction abilities
        - Repetitions: Use verified count of {marker_counts.get('REPETITION', 0)}
          * Categorize types (word, phrase, sound level)
          * Provide examples and count summary
        - Total disfluency assessment: Use verified total of {category_totals['fluency_issues']} 
          * Rate: {category_totals['fluency_issues']/total_words*100:.2f} per 100 words
          * Provide objective rate calculation

        B. Word Retrieval Issues:
        - Circumlocutions: Count and analyze from transcript
        - Incomplete thoughts: Identify abandoned utterances
        - Generic language use: Count vague terms
        - Word-finding efficiency: Assess retrieval success rate

        C. Grammatical Errors (use verified counts):
        - Grammar errors: Use verified count of {marker_counts.get('GRAM_ERROR', 0)}
        - Syntax errors: Use verified count of {marker_counts.get('SYNTAX_ERROR', 0)}
        - Morphological errors: Use verified count of {marker_counts.get('MORPH_ERROR', 0)}
        - Calculate overall grammatical accuracy rate

        2. LANGUAGE SKILLS ASSESSMENT

        A. Vocabulary Analysis (use verified data):
        - Simple vocabulary: Use verified count of {marker_counts.get('SIMPLE_VOCAB', 0)}
        - Complex vocabulary: Use verified count of {marker_counts.get('COMPLEX_VOCAB', 0)}
        - Sophistication ratio: Use verified ratio of {category_totals['vocab_sophistication_ratio']:.3f}
        - Type-Token Ratio: Use verified TTR from basic metrics
        - Provide examples of each vocabulary level from transcript

        B. Grammar and Morphology:
        - Error pattern analysis using verified counts
        - Pattern analysis only
        - Morphological complexity evaluation

        3. COMPLEX SENTENCE ANALYSIS (use verified counts)

        A. Sentence Structure Distribution:
        - Simple sentences: Use verified count of {marker_counts.get('SIMPLE_SENT', 0)}
        - Complex sentences: Use verified count of {marker_counts.get('COMPLEX_SENT', 0)}
        - Compound sentences: Use verified count of {marker_counts.get('COMPOUND_SENT', 0)}
        - Calculate percentages of each type

        B. Syntactic Complexity:
        - MLU analysis: Use verified MLU of {linguistic_metrics.get('mlu_words', 0):.2f} words
        - Average sentence length: Use verified length of {linguistic_metrics.get('avg_sentence_length', 0):.2f} words
        - Subordination and coordination patterns

        4. FIGURATIVE LANGUAGE ANALYSIS
        - Figurative expressions: Use verified count of {marker_counts.get('FIGURATIVE', 0)}
        - Metaphor and idiom identification from transcript
        - Age-appropriate development assessment
        - Abstract language abilities

        5. PRAGMATIC LANGUAGE ASSESSMENT
        - Topic shifts: Use verified count of {marker_counts.get('TOPIC_SHIFT', 0)}
        - Tangential speech: Use verified count of {marker_counts.get('TANGENT', 0)}
        - Coherence breaks: Use verified count of {marker_counts.get('COHERENCE_BREAK', 0)}
        - Referential clarity: Use verified count of {marker_counts.get('PRONOUN_REF', 0)}
        - Overall conversational patterns observed

        6. VOCABULARY AND SEMANTIC ANALYSIS
        - Semantic errors: Use verified count of {marker_counts.get('SEMANTIC_ERROR', 0)}
        - Lexical diversity: Use verified measures from stats summary
        - Word association patterns from transcript analysis
        - Semantic precision and appropriateness

        7. MORPHOLOGICAL AND PHONOLOGICAL ANALYSIS
        - Morphological complexity assessment
        - Derivational and inflectional morphology patterns
        - Error analysis using verified counts
        - Pattern analysis only

        8. QUANTITATIVE METRICS AND NLP FEATURES (use ALL verified data)
        - Total words: {total_words}
        - Total sentences: {linguistic_metrics.get('total_sentences', 0)}
        - Unique words: {linguistic_metrics.get('unique_words', 0)}
        - MLU words: {linguistic_metrics.get('mlu_words', 0):.2f}
        - MLU morphemes: {linguistic_metrics.get('mlu_morphemes', 0):.2f}
        - All error rates and ratios from verified counts



        CRITICAL: Complete ALL 13 sections using verified data and specific transcript examples.
        """
        
        # Get comprehensive analysis
        final_result = call_claude_api_with_continuation(final_prompt)
        return final_result

    def run_full_pipeline(transcript_content, age, gender, slp_notes):
        """Run the complete pipeline but return annotation immediately"""
        if not transcript_content or len(transcript_content.strip()) < 50:
            return "Error: Please provide a longer transcript for analysis.", "", "Error"
        
        # Step 1: Get annotation
        annotated_transcript, annotation_status = run_annotation_step(transcript_content, age, gender, slp_notes)
        
        if annotated_transcript.startswith("Error"):
            return annotated_transcript, "", annotation_status
        
        # Step 2: Run analysis
        analysis_result = run_analysis_step(annotated_transcript, transcript_content, age, gender, slp_notes)
        
        return annotated_transcript, analysis_result, "Complete analysis finished!"
    
    def run_complete_speech_analysis(transcript_content, age, gender, slp_notes):
        """Run the complete speech analysis pipeline with ultimate analysis"""
        if not transcript_content or len(transcript_content.strip()) < 50:
            return "Error: Please provide a longer transcript for analysis.", "", "Error"
        
        # Step 1: Annotate transcript
        annotated_transcript, annotation_status = run_annotation_step(transcript_content, age, gender, slp_notes)
        
        if annotated_transcript.startswith("Error"):
            return annotated_transcript, "", annotation_status
        
        # Step 2: Run ultimate analysis
        ultimate_result = run_ultimate_analysis(annotated_transcript, transcript_content, age, gender, slp_notes)
        
        return annotated_transcript, ultimate_result, "Complete speech analysis finished!"
    
    # Single main event handler
    ultimate_analysis_btn.click(
        fn=run_complete_speech_analysis,
        inputs=[transcript_input, age_input, gender_input, slp_notes_input],
        outputs=[annotated_output, analysis_output, status_display]
    )
    
    annotate_btn.click(
        fn=annotate_transcript,
        inputs=[transcript_input_2, age_input_2, gender_input_2, slp_notes_input_2],
        outputs=[annotation_output]
    )
    

    
    # Quick Questions event handler
    ask_question_btn.click(
        fn=answer_quick_question,
        inputs=[transcript_input_4, question_input, age_input_4, gender_input_4, slp_notes_input_4],
        outputs=[question_output]
    )
    
    # Targeted Analysis event handler
    targeted_analysis_btn.click(
        fn=analyze_targeted_area,
        inputs=[transcript_input_5, analysis_area, age_input_5, gender_input_5, slp_notes_input_5],
        outputs=[targeted_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )