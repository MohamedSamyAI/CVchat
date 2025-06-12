import os
import docx2txt
from langdetect import detect

def extract_cv_content(file_path):
    """
    Extract text content from a CV docx file.
    Strips out formatting and metadata, keeping only relevant content.
    """
    try:
        # Extract text from the docx file
        text = docx2txt.process(file_path)
        
        # Remove extra whitespace and normalize line breaks
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        
        return text
    except Exception as e:
        print(f"Error extracting CV content: {e}")
        return ""

def detect_language(text):
    """
    Detect if the text is in Arabic or English.
    Returns 'ar' for Arabic, 'en' for English, or 'en' as default if detection fails.
    """
    try:
        lang = detect(text)
        # langdetect returns 'ar' for Arabic and 'en' for English
        return lang if lang in ['ar', 'en'] else 'en'
    except:
        # Default to English if detection fails
        return 'en'