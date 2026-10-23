#!/usr/bin/env python3

"""
Advanced features that can be added to the CASL Voice Bot application.
This module contains extensions that SLPs might want to add to the base system.
"""

import os
import pandas as pd
import datetime
import json
from pathlib import Path

class SessionRecorder:
    """Records session data for later analysis and progress tracking"""
    
    def __init__(self, storage_dir="session_data"):
        self.storage_dir = storage_dir
        Path(storage_dir).mkdir(exist_ok=True)
        self.current_session = {
            "timestamp": datetime.datetime.now().isoformat(),
            "student_id": None,
            "transcript": [],
            "assessment": {}
        }
    
    def set_student_id(self, student_id):
        """Set the student ID for the current session"""
        self.current_session["student_id"] = student_id
    
    def add_transcript_entry(self, speaker, text):
        """Add an entry to the transcript"""
        self.current_session["transcript"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "speaker": speaker,
            "text": text
        })
    
    def add_assessment_note(self, category, note):
        """Add an assessment note for a CASL-2 category"""
        if category not in self.current_session["assessment"]:
            self.current_session["assessment"][category] = []
        
        self.current_session["assessment"][category].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "note": note
        })
    
    def save_session(self):
        """Save the current session to a JSON file"""
        student_id = self.current_session["student_id"] or "anonymous"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{student_id}_{timestamp}.json"
        
        with open(os.path.join(self.storage_dir, filename), 'w') as f:
            json.dump(self.current_session, f, indent=2)
        
        return filename


class CASLAnalyzer:
    """Analyzes transcripts based on CASL-2 framework categories"""
    
    def __init__(self):
        self.categories = {
            "lexical_semantic": {
                "description": "Vocabulary knowledge and word meanings",
                "keywords": ["synonym", "antonym", "vocabulary", "word choice", "meaning"]
            },
            "syntactic": {
                "description": "Grammar and sentence structure",
                "keywords": ["grammar", "sentence", "verb tense", "agreement", "structure"]
            },
            "supralinguistic": {
                "description": "Higher-level language skills",
                "keywords": ["inference", "figurative", "metaphor", "context", "implied"]
            },
            "pragmatic": {
                "description": "Social use of language",
                "keywords": ["conversation", "social", "turn-taking", "appropriate", "context"]
            }
        }
    
    def categorize_text(self, text):
        """Categorize text into CASL-2 framework categories"""
        result = {}
        
        for category, info in self.categories.items():
            score = 0
            for keyword in info["keywords"]:
                if keyword.lower() in text.lower():
                    score += 1
            
            if score > 0:
                result[category] = score
        
        return result
    
    def generate_summary(self, transcript):
        """Generate a summary of the transcript based on CASL-2 categories"""
        all_text = " ".join([entry["text"] for entry in transcript])
        categorization = self.categorize_text(all_text)
        
        summary = {
            "categories_covered": list(categorization.keys()),
            "focus_areas": sorted(categorization.items(), key=lambda x: x[1], reverse=True),
            "recommendations": []
        }
        
        # Generate recommendations based on categories covered
        for category in self.categories:
            if category not in categorization:
                summary["recommendations"].append(
                    f"Consider adding more {self.categories[category]['description']} exercises"
                )
        
        return summary


class VoiceMetricsAnalyzer:
    """Analyzes voice metrics for speech patterns"""
    
    def __init__(self):
        self.metrics = {
            "word_count": 0,
            "unique_words": set(),
            "sentence_count": 0,
            "average_words_per_sentence": 0,
            "hesitations": 0,
            "speech_rate": 0  # words per minute
        }
    
    def analyze_text(self, text, duration_seconds=None):
        """Analyze text for speech metrics"""
        # Count words
        words = text.split()
        self.metrics["word_count"] = len(words)
        self.metrics["unique_words"] = set(word.lower() for word in words)
        
        # Count sentences
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        self.metrics["sentence_count"] = len(sentences)
        
        # Calculate average words per sentence
        if self.metrics["sentence_count"] > 0:
            self.metrics["average_words_per_sentence"] = self.metrics["word_count"] / self.metrics["sentence_count"]
        
        # Count hesitations ("um", "uh", "like", etc.)
        hesitation_markers = ["um", "uh", "er", "like", "you know"]
        self.metrics["hesitations"] = sum(1 for word in words if word.lower() in hesitation_markers)
        
        # Calculate speech rate if duration is provided
        if duration_seconds:
            self.metrics["speech_rate"] = (self.metrics["word_count"] / duration_seconds) * 60
        
        return self.metrics
    
    def get_summary(self):
        """Get a summary of the voice metrics analysis"""
        return {
            "word_count": self.metrics["word_count"],
            "vocabulary_diversity": len(self.metrics["unique_words"]) / max(1, self.metrics["word_count"]),
            "average_words_per_sentence": self.metrics["average_words_per_sentence"],
            "hesitation_frequency": self.metrics["hesitations"] / max(1, self.metrics["word_count"]),
            "speech_rate": self.metrics["speech_rate"]
        }


# These classes can be imported and used to extend the base CASL Voice Bot with additional features