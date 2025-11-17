#!/usr/bin/env python3
"""
Flask API Server for Emotion Detection
Run this on your local network or PC
The Android app will send chat messages here
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import os
from collections import defaultdict
import re

app = Flask(__name__)
CORS(app)  # Allow Android app to connect

# ============================================================
# LOAD MODEL ON STARTUP
# ============================================================
class EmotionDetector:
    def __init__(self, model_path='emotion_roberta_model'):
        print("🤖 Loading model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load emotion mapping
        self.emotions = self._load_emotions(model_path)
        self.thresholds = self._load_thresholds(model_path)
        
        print(f"✅ Model loaded with {len(self.emotions)} emotions")
        print(f"Emotions: {', '.join(self.emotions[:10])}...")
    
    def _load_emotions(self, model_path):
        """Load emotion names"""
        # Try emotion_mapping.json
        mapping_path = os.path.join(model_path, 'emotion_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                data = json.load(f)
                if 'emotions' in data:
                    return data['emotions']
        
        # Fallback to default GoEmotions
        return [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def _load_thresholds(self, model_path):
        """Load optimized thresholds"""
        config_path = os.path.join(model_path, 'best_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'thresholds' in config and config['thresholds']:
                    return np.array(config['thresholds'])
        
        # Default threshold
        return np.full(len(self.emotions), 0.5)
    
    def predict_batch(self, texts, top_k=3):
        """Predict emotions for multiple texts"""
        self.model.eval()
        all_results = []
        
        with torch.no_grad():
            # Tokenize
            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
            
            # Process each prediction
            for prob_vector in probs:
                emotions = []
                
                # Get emotions above threshold
                for idx, prob in enumerate(prob_vector):
                    if prob >= self.thresholds[idx]:
                        emotions.append({
                            'emotion': self.emotions[idx],
                            'confidence': float(prob)
                        })
                
                # If no emotions detected, get top K
                if not emotions:
                    top_indices = np.argsort(prob_vector)[-top_k:][::-1]
                    emotions = [
                        {
                            'emotion': self.emotions[idx],
                            'confidence': float(prob_vector[idx])
                        }
                        for idx in top_indices
                    ]
                else:
                    # Sort by confidence and take top K
                    emotions = sorted(emotions, key=lambda x: x['confidence'], reverse=True)[:top_k]
                
                all_results.append(emotions)
        
        return all_results
    
    def analyze_conversation(self, messages, top_k=3):
        """
        Analyze a conversation
        messages: list of {"username": "...", "message": "..."}
        Returns: per-user emotion summary
        """
        if not messages:
            return {'error': 'No messages provided'}
        
        # Extract texts and usernames
        usernames = [msg['username'] for msg in messages]
        texts = [msg['message'] for msg in messages]
        
        # Predict emotions
        predictions = self.predict_batch(texts, top_k=top_k)
        
        # Aggregate by user
        user_emotions = defaultdict(list)
        per_message = []
        
        for msg, preds in zip(messages, predictions):
            username = msg['username']
            message = msg['message']
            
            per_message.append({
                'username': username,
                'message': message,
                'emotions': preds
            })
            
            # Collect emotions for this user
            if preds:
                user_emotions[username].extend([e['emotion'] for e in preds])
        
        # Summarize per user
        user_summary = {}
        for username, emotion_list in user_emotions.items():
            # Count emotions
            emotion_counts = defaultdict(int)
            for emotion in emotion_list:
                emotion_counts[emotion] += 1
            
            # Get top emotions
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [e for e, _ in sorted_emotions[:top_k]]
            
            user_summary[username] = {
                'top_emotions': top_emotions,
                'summary': ', '.join(top_emotions),
                'message_count': len([m for m in messages if m['username'] == username]),
                'emotion_breakdown': dict(sorted_emotions)
            }
        
        return {
            'user_summary': user_summary,
            'per_message': per_message,
            'total_messages': len(messages),
            'total_users': len(user_summary)
        }

# Initialize detector
detector = EmotionDetector()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Check if server is running"""
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'emotions_count': len(detector.emotions)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion for single text
    Body: {"text": "I am so happy today!"}
    """
    try:
        data = request.json
        
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data['text']
        top_k = data.get('top_k', 3)
        
        predictions = detector.predict_batch([text], top_k=top_k)
        
        return jsonify({
            'text': text,
            'emotions': predictions[0]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    """
    Analyze a conversation/chat
    Body: {
        "messages": [
            {"username": "Alice", "message": "I'm so excited!"},
            {"username": "Bob", "message": "That's great news!"},
            ...
        ],
        "top_k": 3
    }
    """
    try:
        data = request.json
        
        if 'messages' not in data:
            return jsonify({'error': 'Missing "messages" field'}), 400
        
        messages = data['messages']
        top_k = data.get('top_k', 3)
        
        if not isinstance(messages, list):
            return jsonify({'error': '"messages" must be a list'}), 400
        
        # Validate message format
        for msg in messages:
            if 'username' not in msg or 'message' not in msg:
                return jsonify({'error': 'Each message must have "username" and "message"'}), 400
        
        # Analyze
        result = detector.analyze_conversation(messages, top_k=top_k)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict emotions for multiple texts
    Body: {
        "texts": ["text1", "text2", ...],
        "top_k": 3
    }
    """
    try:
        data = request.json
        
        if 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field'}), 400
        
        texts = data['texts']
        top_k = data.get('top_k', 3)
        
        if not isinstance(texts, list):
            return jsonify({'error': '"texts" must be a list'}), 400
        
        predictions = detector.predict_batch(texts, top_k=top_k)
        
        results = []
        for text, preds in zip(texts, predictions):
            results.append({
                'text': text,
                'emotions': preds
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 EMOTION DETECTION API SERVER")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /health          - Check server status")
    print("  POST /predict         - Single text prediction")
    print("  POST /analyze_chat    - Analyze conversation")
    print("  POST /batch_predict   - Batch predictions")
    print("\n" + "="*70)
    print("📱 Connect your Android app to: http://YOUR_PC_IP:5000")
    print("="*70 + "\n")
    
    # Run on all interfaces so Android can connect
    # Change to '0.0.0.0' to allow network access
    app.run(host='0.0.0.0', port=5000, debug=False)