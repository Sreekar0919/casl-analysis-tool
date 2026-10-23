#!/usr/bin/env python3

"""
Common utilities for CASL Voice Bot implementations
"""

import os
import time

class CASLAssessment:
    """CASL-2 assessment tracker"""
    
    def __init__(self):
        """Initialize the assessment tracker"""
        self.notes = []
        self.current_assessment = {
            "lexical_semantic": 0,
            "syntactic": 0,
            "supralinguistic": 0,
            "pragmatic": 0
        }
    
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


def save_session_data(transcript, assessment, student_id=None):
    """Save session data to a file"""
    if not student_id:
        student_id = "anonymous"
    
    # Create session data directory if it doesn't exist
    os.makedirs("session_data", exist_ok=True)
    
    # Save transcript
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"session_data/{student_id}_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write("\n".join(transcript))
        f.write("\n\n--- ASSESSMENT NOTES ---\n")
        for note in assessment.notes:
            f.write(f"- {note}\n")
        f.write("\n--- CASL-2 SCORES ---\n")
        for category, score in assessment.current_assessment.items():
            f.write(f"{category.replace('_', ' ').title()}: {score}\n")
    
    return f"Session saved to {filename}"


# Common prompt used by all implementations
CASL_PROMPT = """
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