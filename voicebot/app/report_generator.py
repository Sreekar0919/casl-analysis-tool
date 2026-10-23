#!/usr/bin/env python3

"""
Report generator for CASL Voice Bot.
This module generates assessment reports based on session data.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import jinja2


class CASLReportGenerator:
    """Generates reports from session data"""
    
    def __init__(self, session_data_dir="session_data", reports_dir="reports"):
        """Initialize the report generator"""
        self.session_data_dir = session_data_dir
        self.reports_dir = reports_dir
        
        # Create directories if they don't exist
        Path(session_data_dir).mkdir(exist_ok=True)
        Path(reports_dir).mkdir(exist_ok=True)
        
        # Set up Jinja2 template environment
        self.template_loader = jinja2.FileSystemLoader(searchpath="./templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
    
    def load_session_data(self, filename=None, student_id=None):
        """Load session data from file or by student ID"""
        if filename:
            with open(os.path.join(self.session_data_dir, filename), 'r') as f:
                return json.load(f)
        
        elif student_id:
            # Find all files for this student
            files = [f for f in os.listdir(self.session_data_dir) 
                    if f.startswith(f"{student_id}_") and f.endswith(".json")]
            
            if not files:
                return None
            
            # Sort by date (newest first) and load the most recent
            files.sort(reverse=True)
            with open(os.path.join(self.session_data_dir, files[0]), 'r') as f:
                return json.load(f)
        
        return None
    
    def load_all_student_sessions(self, student_id):
        """Load all sessions for a specific student"""
        files = [f for f in os.listdir(self.session_data_dir) 
                if f.startswith(f"{student_id}_") and f.endswith(".json")]
        
        sessions = []
        for file in sorted(files):
            with open(os.path.join(self.session_data_dir, file), 'r') as f:
                sessions.append(json.load(f))
        
        return sessions
    
    def extract_casl_metrics(self, session_data):
        """Extract CASL-2 metrics from session data"""
        metrics = {
            "lexical_semantic": 0,
            "syntactic": 0,
            "supralinguistic": 0, 
            "pragmatic": 0
        }
        
        # Count assessment notes per category
        assessment = session_data.get("assessment", {})
        for category, notes in assessment.items():
            if category in metrics:
                metrics[category] = len(notes)
        
        return metrics
    
    def generate_progress_chart(self, student_id, output_path=None):
        """Generate a progress chart for a student"""
        sessions = self.load_all_student_sessions(student_id)
        
        if not sessions:
            return None
        
        # Extract dates and metrics
        dates = []
        metrics = {
            "lexical_semantic": [],
            "syntactic": [],
            "supralinguistic": [],
            "pragmatic": []
        }
        
        for session in sessions:
            dates.append(datetime.fromisoformat(session["timestamp"]).strftime("%m/%d/%Y"))
            session_metrics = self.extract_casl_metrics(session)
            
            for category in metrics:
                metrics[category].append(session_metrics.get(category, 0))
        
        # Create chart
        plt.figure(figsize=(10, 6))
        for category, values in metrics.items():
            plt.plot(dates, values, marker='o', label=category.replace('_', ' ').title())
        
        plt.title(f"CASL-2 Assessment Progress for Student {student_id}")
        plt.xlabel("Session Date")
        plt.ylabel("Assessment Score")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or return
        if output_path:
            plt.savefig(output_path)
            return output_path
        else:
            chart_path = os.path.join(self.reports_dir, f"{student_id}_progress.png")
            plt.savefig(chart_path)
            return chart_path
    
    def generate_session_summary(self, session_data):
        """Generate a summary of a single session"""
        if not session_data:
            return None
        
        # Extract basic info
        timestamp = datetime.fromisoformat(session_data["timestamp"])
        student_id = session_data.get("student_id", "anonymous")
        
        # Extract transcript
        transcript = session_data.get("transcript", [])
        
        # Calculate metrics
        word_count = 0
        student_turns = 0
        
        for entry in transcript:
            if entry.get("speaker") == "Student":
                text = entry.get("text", "")
                words = text.split()
                word_count += len(words)
                student_turns += 1
        
        # Get CASL-2 metrics
        casl_metrics = self.extract_casl_metrics(session_data)
        
        # Create summary
        summary = {
            "date": timestamp.strftime("%m/%d/%Y"),
            "time": timestamp.strftime("%H:%M"),
            "student_id": student_id,
            "duration_minutes": len(transcript) // 2,  # Approximate based on turns
            "student_turns": student_turns,
            "total_words": word_count,
            "average_words_per_turn": word_count / max(1, student_turns),
            "casl_metrics": casl_metrics
        }
        
        return summary
    
    def generate_html_report(self, student_id, output_path=None):
        """Generate an HTML report for a student"""
        # Load all sessions for the student
        sessions = self.load_all_student_sessions(student_id)
        
        if not sessions:
            return None
        
        # Generate progress chart
        chart_path = self.generate_progress_chart(student_id)
        
        # Get latest session data
        latest_session = sessions[-1]
        latest_summary = self.generate_session_summary(latest_session)
        
        # Calculate overall progress
        if len(sessions) > 1:
            first_metrics = self.extract_casl_metrics(sessions[0])
            latest_metrics = self.extract_casl_metrics(sessions[-1])
            
            progress = {}
            for category in first_metrics:
                if first_metrics[category] > 0:
                    progress[category] = (latest_metrics[category] - first_metrics[category]) / first_metrics[category]
                else:
                    progress[category] = 0 if latest_metrics[category] == 0 else 1
        else:
            progress = {category: 0 for category in latest_summary["casl_metrics"]}
        
        # Prepare report data
        report_data = {
            "student_id": student_id,
            "report_date": datetime.now().strftime("%m/%d/%Y"),
            "session_count": len(sessions),
            "latest_session": latest_summary,
            "progress": progress,
            "chart_path": os.path.basename(chart_path),
            "recommendations": self.generate_recommendations(sessions)
        }
        
        # Load and render template
        try:
            template = self.template_env.get_template("report_template.html")
            report_html = template.render(**report_data)
            
            # Save report
            if not output_path:
                output_path = os.path.join(self.reports_dir, f"{student_id}_report.html")
            
            with open(output_path, 'w') as f:
                f.write(report_html)
            
            return output_path
            
        except jinja2.exceptions.TemplateNotFound:
            # Create a simple report if template is not found
            report = f"CASL-2 Assessment Report for Student {student_id}\n"
            report += f"Report Date: {report_data['report_date']}\n"
            report += f"Total Sessions: {report_data['session_count']}\n\n"
            
            report += "Latest Session Summary:\n"
            for key, value in latest_summary.items():
                if key != "casl_metrics":
                    report += f"  {key}: {value}\n"
            
            report += "\nCASL-2 Metrics:\n"
            for category, value in latest_summary["casl_metrics"].items():
                report += f"  {category}: {value}\n"
            
            report += "\nRecommendations:\n"
            for rec in report_data["recommendations"]:
                report += f"  - {rec}\n"
            
            # Save simple report
            if not output_path:
                output_path = os.path.join(self.reports_dir, f"{student_id}_report.txt")
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            return output_path
    
    def generate_recommendations(self, sessions):
        """Generate recommendations based on session data"""
        if not sessions:
            return []
        
        latest_session = sessions[-1]
        metrics = self.extract_casl_metrics(latest_session)
        
        recommendations = []
        
        # Check for areas needing improvement
        weak_areas = [category for category, value in metrics.items() if value < 2]
        for area in weak_areas:
            if area == "lexical_semantic":
                recommendations.append("Focus on vocabulary building exercises such as synonyms, antonyms, and word associations")
            elif area == "syntactic":
                recommendations.append("Practice sentence formation and grammar through structured activities")
            elif area == "supralinguistic":
                recommendations.append("Work on understanding figurative language and making inferences from context")
            elif area == "pragmatic":
                recommendations.append("Engage in role-playing activities to practice social communication skills")
        
        # Add general recommendations
        if len(sessions) > 1:
            recommendations.append("Continue regular assessment sessions to track progress")
        
        if not recommendations:
            recommendations.append("Continue current therapy approach as all areas show adequate progress")
        
        return recommendations


# This module can be used to generate reports from the session data collected by the CASL Voice Bot
if __name__ == "__main__":
    # Example usage
    report_gen = CASLReportGenerator()
    # report_gen.generate_html_report("student123")