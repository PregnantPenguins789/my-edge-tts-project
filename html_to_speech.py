#!/usr/bin/env python3
"""
HTML to Speech Converter
Extracts text from HTML and converts it to speech using edge-tts

USAGE:
======
1. Save this script as: html_to_speech.py
2. Make executable: chmod +x html_to_speech.py
3. Run with your HTML file:
   python3 html_to_speech.py mypage.html output.mp3

EXAMPLES:
=========
  # Convert HTML to MP3
  python3 html_to_speech.py index.html speech.mp3
  
  # Convert with specific voice
  python3 html_to_speech.py index.html speech.mp3 "en-US-AriaNeural"
  
  # Stream to speakers (no file output)
  python3 html_to_speech.py index.html

VOICE OPTIONS:
==============
List all available voices:
  edge-tts --list-voices

Common voices:
  - en-US-EmmaMultilingualNeural (default)
  - en-US-AriaNeural
  - en-US-GuyNeural
  - en-GB-SoniaNeural
  - en-AU-NatashaNeural

REQUIREMENTS:
=============
- Python 3.6+
- edge-tts installed in virtual environment (pip install edge-tts)
"""

import sys
import subprocess
from html.parser import HTMLParser
from pathlib import Path

class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, ignoring tags and scripts"""
    
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False
    
    def handle_starttag(self, tag, attrs):
        # Skip script and style content
        if tag in ('script', 'style'):
            self.skip = True
    
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = False
        # Add line break for paragraph-like elements
        elif tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'li'):
            self.text.append('\n')
    
    def handle_data(self, data):
        if not self.skip:
            # Clean up whitespace but preserve structure
            cleaned = data.strip()
            if cleaned:
                self.text.append(cleaned + ' ')
    
    def get_text(self):
        return ''.join(self.text).strip()

def html_to_speech(html_file, output_file=None, voice=None):
    """
    Convert HTML file to speech
    
    Args:
        html_file: Path to HTML file
        output_file: Optional output audio file (MP3)
        voice: Optional voice selection
    """
    
    # Read and parse HTML file
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{html_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Extract text from HTML
    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    text = extractor.get_text()
    
    if not text:
        print("Error: No text content found in HTML file")
        sys.exit(1)
    
    print(f"Extracted text ({len(text)} characters):")
    print("-" * 50)
    print(text[:200] + "..." if len(text) > 200 else text)
    print("-" * 50)
    
    # Build edge-tts command
    cmd = ['edge-tts', '--text', text]
    
    if voice:
        cmd.extend(['--voice', voice])
    
    if output_file:
        cmd.extend(['--write-media', output_file])
        print(f"Generating audio... (output: {output_file})")
    else:
        print("Generating audio and playing...")
    
    # Run edge-tts
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        
        if output_file:
            print(f"✓ Audio saved to: {output_file}")
        else:
            print("✓ Done")
    
    except FileNotFoundError:
        print("Error: edge-tts not found. Make sure it's installed: pip install edge-tts")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 html_to_speech.py <html_file> [output_file] [voice]")
        print()
        print("Examples:")
        print("  python3 html_to_speech.py mypage.html")
        print("  python3 html_to_speech.py mypage.html output.mp3")
        print("  python3 html_to_speech.py mypage.html output.mp3 'en-US-AriaNeural'")
        print()
        print("To list available voices, run: edge-tts --list-voices")
        sys.exit(1)
    
    html_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    voice = sys.argv[3] if len(sys.argv) > 3 else None
    
    html_to_speech(html_file, output_file, voice)