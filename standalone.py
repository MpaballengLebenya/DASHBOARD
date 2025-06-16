#!/usr/bin/env python3
"""
Standalone Sentiment Analysis Dashboard
No external dependencies required - works with just Python standard library
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import json
import csv
import re
from datetime import datetime
from collections import Counter

class SentimentAnalyzer:
    def __init__(self):
        # Enhanced word lists for better accuracy
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'awesome',
            'brilliant', 'outstanding', 'superb', 'magnificent', 'perfect', 'beautiful',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'delighted',
            'thrilled', 'excited', 'impressed', 'remarkable', 'incredible', 'best',
            'fabulous', 'marvelous', 'terrific', 'splendid', 'phenomenal', 'spectacular'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'dislike',
            'disgusting', 'pathetic', 'ridiculous', 'stupid', 'boring', 'annoying',
            'frustrating', 'disappointing', 'useless', 'worthless', 'poor', 'inadequate',
            'mediocre', 'unsatisfactory', 'unacceptable', 'inferior', 'defective',
            'faulty', 'broken', 'damaged', 'ugly', 'unpleasant', 'disturbing'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
            'completely': 1.7, 'totally': 1.6, 'really': 1.4, 'quite': 1.2,
            'pretty': 1.1, 'somewhat': 0.8, 'rather': 0.9, 'fairly': 0.9
        }
        
        self.negators = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', 'none'}
    
    def preprocess_text(self, text):
        """Clean and prepare text for analysis"""
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove punctuation except for sentence boundaries
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with enhanced accuracy"""
        if not text.strip():
            return {'label': 'NEUTRAL', 'score': 0.5, 'explanation': 'Empty text'}
        
        words = self.preprocess_text(text)
        total_words = len(words)
        
        if total_words == 0:
            return {'label': 'NEUTRAL', 'score': 0.5, 'explanation': 'No valid words found'}
        
        positive_score = 0
        negative_score = 0
        sentiment_words = []
        
        for i, word in enumerate(words):
            # Check for negation in previous 2 words
            negated = any(neg in words[max(0, i-2):i] for neg in self.negators)
            
            # Check for intensifiers in previous 2 words
            intensifier = 1.0
            for j in range(max(0, i-2), i):
                if words[j] in self.intensifiers:
                    intensifier = self.intensifiers[words[j]]
                    break
            
            if word in self.positive_words:
                score = intensifier
                if negated:
                    negative_score += score
                    sentiment_words.append(f"NOT {word}")
                else:
                    positive_score += score
                    sentiment_words.append(word)
            
            elif word in self.negative_words:
                score = intensifier
                if negated:
                    positive_score += score
                    sentiment_words.append(f"NOT {word}")
                else:
                    negative_score += score
                    sentiment_words.append(word)
        
        # Calculate final sentiment
        total_sentiment_score = positive_score + negative_score
        
        if total_sentiment_score == 0:
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'explanation': 'No sentiment words detected'
            }
        
        # Normalize scores
        positive_ratio = positive_score / total_sentiment_score
        negative_ratio = negative_score / total_sentiment_score
        
        # Determine sentiment
        if positive_ratio > negative_ratio:
            confidence = min(0.95, 0.5 + (positive_ratio - negative_ratio))
            label = 'POSITIVE'
        elif negative_ratio > positive_ratio:
            confidence = min(0.95, 0.5 + (negative_ratio - positive_ratio))
            label = 'NEGATIVE'
        else:
            confidence = 0.5
            label = 'NEUTRAL'
        
        explanation = f"Found {len(sentiment_words)} sentiment indicators: {', '.join(sentiment_words[:5])}"
        if len(sentiment_words) > 5:
            explanation += "..."
        
        return {
            'label': label,
            'score': confidence,
            'explanation': explanation
        }
    
    def extract_keywords(self, text, max_keywords=10):
        """Extract meaningful keywords from text"""
        words = self.preprocess_text(text)
        
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Filter meaningful words
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        
        # Count frequency and return most common
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(max_keywords)]

class SentimentDashboard:
    def __init__(self, root):
        self.root = root
        self.analyzer = SentimentAnalyzer()
        self.results = []
        
        # Modern color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'background': '#ffffff',
            'sidebar': '#f0f2f6',
            'text': '#262730'
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        self.root.title("📊 Sentiment Analysis Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg=self.colors['background'])
        
        # Configure modern styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=self.colors['primary'])
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['text'])
        style.configure('Custom.TNotebook', tabposition='n')
        style.configure('Custom.TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10))
        style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('Success.TButton', foreground='white')
        style.configure('Danger.TButton', foreground='white')
        
        # Main container with padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="📊 Sentiment Analysis Dashboard", style='Title.TLabel')
        title_label.pack(side='left')
        
        subtitle_label = ttk.Label(header_frame, text="Analyze sentiment in text data with multi-class classification and interactive features", 
                                  font=('Segoe UI', 10), foreground='gray')
        subtitle_label.pack(side='left', padx=(20, 0))
        
        # Create modern notebook for tabs
        self.notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="📝 Text Analysis")
        self.setup_analysis_tab()
        
        # Results Tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📊 Results & Visualizations")
        self.setup_results_tab()
        
        # Batch Tab
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="📈 Batch Processing")
        self.setup_batch_tab()
        
        # Settings Tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ Settings")
        self.setup_settings_tab()
    
    def setup_analysis_tab(self):
        # Create main layout with left and right panels
        main_paned = ttk.PanedWindow(self.analysis_frame, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Left panel for input
        left_frame = ttk.LabelFrame(main_paned, text="📝 Text Input", padding=15)
        main_paned.add(left_frame, weight=1)
        
        # Input method selection
        input_method_frame = ttk.Frame(left_frame)
        input_method_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(input_method_frame, text="Input Method:", style='Heading.TLabel').pack(anchor='w')
        self.input_method = tk.StringVar(value="direct")
        
        method_frame = ttk.Frame(input_method_frame)
        method_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(method_frame, text="Direct Text Entry", variable=self.input_method, 
                       value="direct", command=self.toggle_input_method).pack(side='left', padx=(0, 20))
        ttk.Radiobutton(method_frame, text="File Upload", variable=self.input_method, 
                       value="file", command=self.toggle_input_method).pack(side='left')
        
        # Text input area
        text_label_frame = ttk.Frame(left_frame)
        text_label_frame.pack(fill='x', pady=(10, 5))
        ttk.Label(text_label_frame, text="Enter text to analyze:", style='Heading.TLabel').pack(anchor='w')
        
        self.text_input = scrolledtext.ScrolledText(left_frame, height=12, width=50, 
                                                   font=('Segoe UI', 10), wrap=tk.WORD)
        self.text_input.pack(fill='both', expand=True, pady=(0, 10))
        
        # File upload frame (initially hidden)
        self.file_frame = ttk.Frame(left_frame)
        
        file_label = ttk.Label(self.file_frame, text="Selected file:", style='Heading.TLabel')
        file_label.pack(anchor='w', pady=(0, 5))
        
        self.file_label = ttk.Label(self.file_frame, text="No file selected", foreground='gray')
        self.file_label.pack(anchor='w', pady=(0, 10))
        
        # Action buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill='x', pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="🔍 Analyze Sentiment", 
                                     command=self.analyze_text, style='Primary.TButton')
        self.analyze_btn.pack(side='left', padx=(0, 10))
        
        self.load_btn = ttk.Button(button_frame, text="📁 Load File", command=self.load_file)
        self.load_btn.pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_input).pack(side='left')
        
        # Right panel for results
        right_frame = ttk.LabelFrame(main_paned, text="📊 Analysis Results", padding=15)
        main_paned.add(right_frame, weight=1)
        
        # Results display with better formatting
        self.results_display = scrolledtext.ScrolledText(right_frame, height=20, width=50,
                                                        font=('Consolas', 10), wrap=tk.WORD)
        self.results_display.pack(fill='both', expand=True)
        
        # Configure text tags for colored output
        self.results_display.tag_configure('header', font=('Segoe UI', 12, 'bold'), foreground=self.colors['primary'])
        self.results_display.tag_configure('positive', foreground=self.colors['success'], font=('Segoe UI', 10, 'bold'))
        self.results_display.tag_configure('negative', foreground=self.colors['danger'], font=('Segoe UI', 10, 'bold'))
        self.results_display.tag_configure('neutral', foreground='gray', font=('Segoe UI', 10, 'bold'))
        self.results_display.tag_configure('label', font=('Segoe UI', 10, 'bold'))
        self.results_display.tag_configure('value', font=('Segoe UI', 10))
        
        # Initialize with welcome message
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Display welcome message in results area"""
        welcome_text = """
📊 SENTIMENT ANALYSIS DASHBOARD
═══════════════════════════════════════════════════════

Welcome! This tool analyzes the emotional tone of text using advanced 
sentiment classification algorithms.

Features:
• Multi-class sentiment analysis (Positive, Negative, Neutral)
• Confidence scoring for each classification
• Keyword extraction to identify sentiment drivers
• Batch processing for multiple texts
• Export results in CSV and JSON formats

To get started:
1. Enter your text in the input area or upload a file
2. Click "Analyze Sentiment" to process
3. View detailed results and statistics

Ready to analyze your first text!
        """
        self.results_display.insert(tk.END, welcome_text)
    
    def toggle_input_method(self):
        """Toggle between direct input and file upload"""
        if self.input_method.get() == "file":
            self.text_input.pack_forget()
            self.file_frame.pack(fill='x', pady=(10, 0))
            self.load_btn.configure(state='normal')
        else:
            self.file_frame.pack_forget()
            self.text_input.pack(fill='both', expand=True, pady=(0, 10))
            self.load_btn.configure(state='disabled')
    
    def setup_results_tab(self):
        # Main container with padding
        main_container = ttk.Frame(self.results_frame)
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Statistics cards at top
        stats_container = ttk.Frame(main_container)
        stats_container.pack(fill='x', pady=(0, 15))
        
        ttk.Label(stats_container, text="📈 Analytics Overview", style='Heading.TLabel').pack(anchor='w', pady=(0, 10))
        
        # Statistics cards
        cards_frame = ttk.Frame(stats_container)
        cards_frame.pack(fill='x')
        
        self.stats_labels = {}
        stat_configs = [
            ('Total', '📊', self.colors['primary']),
            ('Positive', '😊', self.colors['success']), 
            ('Negative', '😞', self.colors['danger']),
            ('Neutral', '😐', 'gray')
        ]
        
        for i, (stat, emoji, color) in enumerate(stat_configs):
            card = ttk.LabelFrame(cards_frame, text=f"{emoji} {stat}", padding=10)
            card.pack(side='left', fill='x', expand=True, padx=(0, 10) if i < 3 else (0, 0))
            
            value_label = ttk.Label(card, text="0", font=('Segoe UI', 20, 'bold'), foreground=color)
            value_label.pack()
            
            self.stats_labels[stat] = value_label
        
        # Results table with modern styling
        table_frame = ttk.LabelFrame(main_container, text="📋 Analysis Results", padding=10)
        table_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Table
        columns = ('Text Preview', 'Sentiment', 'Confidence', 'Keywords', 'Timestamp')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        # Configure columns
        column_configs = {
            'Text Preview': 300,
            'Sentiment': 100,
            'Confidence': 100,
            'Keywords': 200,
            'Timestamp': 150
        }
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=column_configs.get(col, 100))
        
        # Scrollbar for table
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Action buttons
        button_frame = ttk.Frame(main_container)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Label(button_frame, text="📤 Export & Actions", style='Heading.TLabel').pack(anchor='w', pady=(0, 10))
        
        actions_frame = ttk.Frame(button_frame)
        actions_frame.pack(fill='x')
        
        ttk.Button(actions_frame, text="📊 Export CSV", command=self.export_csv, 
                  style='Primary.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(actions_frame, text="📄 Export JSON", command=self.export_json, 
                  style='Primary.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(actions_frame, text="📈 View Statistics", command=self.show_detailed_stats).pack(side='left', padx=(0, 10))
        ttk.Button(actions_frame, text="🗑️ Clear Results", command=self.clear_results, 
                  style='Danger.TButton').pack(side='right')
    
    def setup_batch_tab(self):
        # Main container
        main_container = ttk.Frame(self.batch_frame)
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header
        ttk.Label(main_container, text="📈 Batch Text Processing", style='Title.TLabel').pack(anchor='w', pady=(0, 5))
        ttk.Label(main_container, text="Process multiple texts simultaneously for bulk sentiment analysis", 
                 font=('Segoe UI', 10), foreground='gray').pack(anchor='w', pady=(0, 20))
        
        # Input section
        input_frame = ttk.LabelFrame(main_container, text="📝 Batch Input", padding=15)
        input_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Instructions
        instructions = ttk.Label(input_frame, text="Enter multiple texts (one per line) or load from file:", 
                                style='Heading.TLabel')
        instructions.pack(anchor='w', pady=(0, 10))
        
        # Text area
        self.batch_input = scrolledtext.ScrolledText(input_frame, height=12, font=('Segoe UI', 10), wrap=tk.WORD)
        self.batch_input.pack(fill='both', expand=True, pady=(0, 15))
        
        # Sample data button
        sample_frame = ttk.Frame(input_frame)
        sample_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(sample_frame, text="📋 Load Sample Data", command=self.load_sample_data).pack(side='left')
        ttk.Label(sample_frame, text="Load example texts to test batch processing", 
                 font=('Segoe UI', 9), foreground='gray').pack(side='left', padx=(10, 0))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_container, text="🎛️ Processing Controls", padding=15)
        control_frame.pack(fill='x', pady=(0, 15))
        
        # Buttons row
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill='x')
        
        ttk.Button(buttons_frame, text="🚀 Process Batch", command=self.process_batch, 
                  style='Primary.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="📁 Load Batch File", command=self.load_batch_file).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="🗑️ Clear Input", command=self.clear_batch_input).pack(side='left')
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_container, text="📊 Processing Progress", padding=15)
        progress_frame.pack(fill='x')
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to process batch", font=('Segoe UI', 10))
        self.progress_label.pack(anchor='w', pady=(0, 5))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill='x')
    
    def setup_settings_tab(self):
        # Main container
        main_container = ttk.Frame(self.settings_frame)
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header
        ttk.Label(main_container, text="⚙️ Application Settings", style='Title.TLabel').pack(anchor='w', pady=(0, 5))
        ttk.Label(main_container, text="Configure analysis parameters and application preferences", 
                 font=('Segoe UI', 10), foreground='gray').pack(anchor='w', pady=(0, 20))
        
        # Analysis Settings
        analysis_frame = ttk.LabelFrame(main_container, text="🔍 Analysis Configuration", padding=15)
        analysis_frame.pack(fill='x', pady=(0, 15))
        
        # Confidence threshold
        threshold_frame = ttk.Frame(analysis_frame)
        threshold_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Confidence Threshold:", style='Heading.TLabel').pack(anchor='w')
        ttk.Label(threshold_frame, text="Minimum confidence score for reliable classifications", 
                 font=('Segoe UI', 9), foreground='gray').pack(anchor='w', pady=(0, 5))
        
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, 
                                   orient='horizontal', length=300)
        threshold_scale.pack(anchor='w')
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.50")
        self.threshold_label.pack(anchor='w')
        threshold_scale.configure(command=self.update_threshold_label)
        
        # Max keywords
        keywords_frame = ttk.Frame(analysis_frame)
        keywords_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(keywords_frame, text="Maximum Keywords:", style='Heading.TLabel').pack(anchor='w')
        ttk.Label(keywords_frame, text="Number of keywords to extract from each text", 
                 font=('Segoe UI', 9), foreground='gray').pack(anchor='w', pady=(0, 5))
        
        self.max_keywords = tk.IntVar(value=10)
        keywords_spinbox = ttk.Spinbox(keywords_frame, from_=1, to=20, textvariable=self.max_keywords, width=10)
        keywords_spinbox.pack(anchor='w')
        
        # Application Settings
        app_frame = ttk.LabelFrame(main_container, text="🎨 Application Preferences", padding=15)
        app_frame.pack(fill='x', pady=(0, 15))
        
        # Auto-save results
        self.auto_save = tk.BooleanVar(value=False)
        auto_save_check = ttk.Checkbutton(app_frame, text="Auto-save results", variable=self.auto_save)
        auto_save_check.pack(anchor='w', pady=(0, 5))
        
        # Show detailed explanations
        self.show_explanations = tk.BooleanVar(value=True)
        explanations_check = ttk.Checkbutton(app_frame, text="Show detailed explanations", variable=self.show_explanations)
        explanations_check.pack(anchor='w', pady=(0, 5))
        
        # About section
        about_frame = ttk.LabelFrame(main_container, text="ℹ️ About", padding=15)
        about_frame.pack(fill='x')
        
        about_text = """Sentiment Analysis Dashboard v2.0
Built with Python and tkinter for desktop sentiment analysis.
Uses advanced word-based classification algorithms.

Features: Multi-class sentiment analysis, keyword extraction, 
batch processing, and comprehensive export capabilities."""
        
        ttk.Label(about_frame, text=about_text, font=('Segoe UI', 9), foreground=self.colors['text']).pack(anchor='w')
    
    def analyze_text(self):
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze")
            return
        
        result = self.analyzer.analyze_sentiment(text)
        keywords = self.analyzer.extract_keywords(text)
        
        # Store result
        analysis_result = {
            'text': text,
            'sentiment': result['label'],
            'confidence': result['score'],
            'explanation': result['explanation'],
            'keywords': keywords,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.results.append(analysis_result)
        
        # Display result
        self.display_result(analysis_result)
        self.update_results_table()
        self.update_statistics()
    
    def display_result(self, result):
        self.results_display.delete(1.0, tk.END)
        
        # Header with emoji and formatting
        self.results_display.insert(tk.END, "📊 SENTIMENT ANALYSIS RESULTS\n", 'header')
        self.results_display.insert(tk.END, "═" * 60 + "\n\n")
        
        # Text preview
        text_preview = result['text'][:300] + '...' if len(result['text']) > 300 else result['text']
        self.results_display.insert(tk.END, "📝 Text: ", 'label')
        self.results_display.insert(tk.END, f"{text_preview}\n\n", 'value')
        
        # Sentiment with color coding
        sentiment_emoji = {'POSITIVE': '😊', 'NEGATIVE': '😞', 'NEUTRAL': '😐'}
        emoji = sentiment_emoji.get(result['sentiment'], '🤔')
        
        self.results_display.insert(tk.END, f"{emoji} Sentiment: ", 'label')
        tag = result['sentiment'].lower()
        self.results_display.insert(tk.END, f"{result['sentiment']}\n", tag)
        
        # Confidence with visual indicator
        confidence_pct = result['confidence'] * 100
        confidence_bar = "█" * int(confidence_pct / 10) + "░" * (10 - int(confidence_pct / 10))
        
        self.results_display.insert(tk.END, "📈 Confidence: ", 'label')
        self.results_display.insert(tk.END, f"{confidence_pct:.1f}% [{confidence_bar}]\n", 'value')
        
        # Explanation
        if self.show_explanations.get() if hasattr(self, 'show_explanations') else True:
            self.results_display.insert(tk.END, "💡 Explanation: ", 'label')
            self.results_display.insert(tk.END, f"{result['explanation']}\n\n", 'value')
        
        # Keywords
        if result['keywords']:
            self.results_display.insert(tk.END, "🔑 Keywords: ", 'label')
            max_kw = self.max_keywords.get() if hasattr(self, 'max_keywords') else 10
            keywords_text = ', '.join([f"'{kw}'" for kw in result['keywords'][:max_kw]])
            self.results_display.insert(tk.END, f"{keywords_text}\n\n", 'value')
        
        # Threshold check
        threshold = self.confidence_threshold.get() if hasattr(self, 'confidence_threshold') else 0.5
        threshold_met = "✅ RELIABLE" if result['confidence'] >= threshold else "⚠️ LOW CONFIDENCE"
        self.results_display.insert(tk.END, "🎯 Reliability: ", 'label')
        self.results_display.insert(tk.END, f"{threshold_met} (threshold: {threshold:.1f})\n\n", 'value')
        
        # Timestamp
        self.results_display.insert(tk.END, "⏰ Analyzed: ", 'label')
        self.results_display.insert(tk.END, f"{result['timestamp']}\n", 'value')
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_input.delete(1.0, tk.END)
                    self.text_input.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {str(e)}")
    
    def clear_input(self):
        self.text_input.delete(1.0, tk.END)
        self.results_display.delete(1.0, tk.END)
    
    def process_batch(self):
        batch_text = self.batch_input.get(1.0, tk.END).strip()
        if not batch_text:
            messagebox.showwarning("Warning", "Please enter texts to process")
            return
        
        texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
        if not texts:
            messagebox.showwarning("Warning", "No valid texts found")
            return
        
        self.progress['maximum'] = len(texts)
        self.progress['value'] = 0
        
        for i, text in enumerate(texts):
            result = self.analyzer.analyze_sentiment(text)
            keywords = self.analyzer.extract_keywords(text)
            
            analysis_result = {
                'text': text,
                'sentiment': result['label'],
                'confidence': result['score'],
                'explanation': result['explanation'],
                'keywords': keywords,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results.append(analysis_result)
            self.progress['value'] = i + 1
            self.root.update_idletasks()
        
        self.update_results_table()
        self.update_statistics()
        messagebox.showinfo("Complete", f"Processed {len(texts)} texts successfully!")
    
    def load_batch_file(self):
        file_path = filedialog.askopenfilename(
            title="Select batch file",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    if file_path.endswith('.csv'):
                        # Try to read CSV and extract text from first column
                        content = file.read()
                        lines = content.strip().split('\n')
                        texts = []
                        for line in lines[1:]:  # Skip header
                            if line.strip():
                                # Simple CSV parsing (assumes no commas in text)
                                first_field = line.split(',')[0].strip().strip('"')
                                if first_field:
                                    texts.append(first_field)
                        content = '\n'.join(texts)
                    else:
                        content = file.read()
                    
                    self.batch_input.delete(1.0, tk.END)
                    self.batch_input.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {str(e)}")
    
    def update_results_table(self):
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add results
        for result in self.results:
            preview = result['text'][:50] + '...' if len(result['text']) > 50 else result['text']
            confidence = f"{result['confidence']:.1%}"
            keywords = ', '.join(result['keywords'][:3]) if result['keywords'] else 'None'
            
            self.results_tree.insert('', 'end', values=(
                preview,
                result['sentiment'],
                confidence,
                keywords,
                result['timestamp']
            ))
    
    def update_statistics(self):
        if not self.results:
            return
        
        total = len(self.results)
        positive = sum(1 for r in self.results if r['sentiment'] == 'POSITIVE')
        negative = sum(1 for r in self.results if r['sentiment'] == 'NEGATIVE')
        neutral = sum(1 for r in self.results if r['sentiment'] == 'NEUTRAL')
        
        self.stats_labels['Total'].config(text=f"{total}")
        self.stats_labels['Positive'].config(text=f"{positive}")
        self.stats_labels['Negative'].config(text=f"{negative}")
        self.stats_labels['Neutral'].config(text=f"{neutral}")
    
    def show_detailed_stats(self):
        """Show detailed statistics in a popup window"""
        if not self.results:
            messagebox.showinfo("No Data", "No analysis results available for statistics.")
            return
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 Detailed Statistics")
        stats_window.geometry("600x400")
        stats_window.configure(bg=self.colors['background'])
        
        # Calculate detailed statistics
        total = len(self.results)
        positive = sum(1 for r in self.results if r['sentiment'] == 'POSITIVE')
        negative = sum(1 for r in self.results if r['sentiment'] == 'NEGATIVE')
        neutral = sum(1 for r in self.results if r['sentiment'] == 'NEUTRAL')
        
        avg_confidence = sum(r['confidence'] for r in self.results) / total
        high_confidence = sum(1 for r in self.results if r['confidence'] >= 0.8)
        
        # All keywords
        all_keywords = []
        for result in self.results:
            all_keywords.extend(result['keywords'])
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(10)
        
        # Create stats display
        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD, font=('Consolas', 10))
        stats_text.pack(fill='both', expand=True, padx=20, pady=20)
        
        stats_content = f"""
📊 DETAILED ANALYSIS STATISTICS
═══════════════════════════════════════════════════════

📈 SENTIMENT DISTRIBUTION
• Total Analyses: {total}
• Positive: {positive} ({positive/total*100:.1f}%)
• Negative: {negative} ({negative/total*100:.1f}%)
• Neutral: {neutral} ({neutral/total*100:.1f}%)

📊 CONFIDENCE METRICS
• Average Confidence: {avg_confidence:.1%}
• High Confidence (>80%): {high_confidence} ({high_confidence/total*100:.1f}%)
• Reliability Score: {'Excellent' if avg_confidence > 0.8 else 'Good' if avg_confidence > 0.6 else 'Fair'}

🔑 TOP KEYWORDS
{chr(10).join([f"• {word}: {count} occurrences" for word, count in top_keywords[:10]])}

📋 ANALYSIS SUMMARY
• Most common sentiment: {max(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], key=lambda x: sum(1 for r in self.results if r['sentiment'] == x))}
• Confidence range: {min(r['confidence'] for r in self.results):.1%} - {max(r['confidence'] for r in self.results):.1%}
• Total keywords extracted: {len(all_keywords)}
• Unique keywords: {len(set(all_keywords))}
        """
        
        stats_text.insert(tk.END, stats_content)
        stats_text.configure(state='disabled')
    
    def load_sample_data(self):
        """Load sample texts for demonstration"""
        sample_texts = [
            "I absolutely love this product! It's amazing and works perfectly.",
            "This is the worst experience I've ever had. Completely disappointed.",
            "The service is okay, nothing special but it works as expected.",
            "Fantastic quality and great customer support. Highly recommend!",
            "Not satisfied with the purchase. Poor quality and overpriced.",
            "Average performance, meets basic requirements but lacks innovation.",
            "Excellent value for money and fast delivery. Very happy!",
            "Terrible customer service and low quality. Avoid at all costs.",
            "Good product overall, does what it's supposed to do.",
            "Outstanding design and performance. Worth every penny!"
        ]
        
        self.batch_input.delete(1.0, tk.END)
        self.batch_input.insert(1.0, '\n'.join(sample_texts))
        messagebox.showinfo("Sample Data Loaded", f"Loaded {len(sample_texts)} sample texts for batch processing.")
    
    def clear_batch_input(self):
        """Clear the batch input area"""
        if messagebox.askyesno("Confirm", "Clear all batch input text?"):
            self.batch_input.delete(1.0, tk.END)
    
    def update_threshold_label(self, value):
        """Update the threshold label when slider changes"""
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def export_csv(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Text', 'Sentiment', 'Confidence', 'Keywords', 'Timestamp'])
                    
                    for result in self.results:
                        writer.writerow([
                            result['text'],
                            result['sentiment'],
                            f"{result['confidence']:.3f}",
                            ', '.join(result['keywords']),
                            result['timestamp']
                        ])
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export CSV: {str(e)}")
    
    def export_json(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save JSON file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(self.results, file, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export JSON: {str(e)}")
    
    def clear_results(self):
        if messagebox.askyesno("Confirm", "Clear all results?"):
            self.results.clear()
            self.update_results_table()
            self.update_statistics()
            self.results_display.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    app = SentimentDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()