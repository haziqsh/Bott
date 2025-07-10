"""
Advanced AI Models for Trading Analysis
Includes: HuggingFace Transformers, Sentiment Analysis, Pattern Recognition, Ensemble Models
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIModels:
    """Advanced AI Models for Trading Analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.pattern_recognizer = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        # Initialize sentiment analysis models
        self.init_sentiment_models()
        
        # Initialize pattern recognition models
        self.init_pattern_models()
        
        # Initialize ensemble models
        self.init_ensemble_models()
    
    def init_sentiment_models(self):
        """Initialize HuggingFace sentiment analysis models"""
        try:
            # Financial-specific sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Backup general sentiment model
            self.general_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            print("✅ Sentiment analysis models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing sentiment models: {e}")
            # Fallback to basic sentiment analysis
            self.sentiment_analyzer = pipeline("sentiment-analysis", return_all_scores=True)
    
    def init_pattern_models(self):
        """Initialize pattern recognition models"""
        try:
            # LSTM for pattern recognition
            self.pattern_lstm = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 5)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  # Buy, Sell, Hold
            ])
            
            self.pattern_lstm.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Pattern recognition models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing pattern models: {e}")
            self.pattern_lstm = None
    
    def init_ensemble_models(self):
        """Initialize ensemble learning models"""
        try:
            # Create ensemble of different algorithms
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            
            # Voting classifier ensemble
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model),
                    ('xgb', xgb_model),
                    ('lgb', lgb_model)
                ],
                voting='soft'
            )
            
            print("✅ Ensemble models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing ensemble models: {e}")
            self.ensemble_model = None
    
    def analyze_sentiment(self, text: str):
        """Analyze sentiment of financial text"""
        try:
            if not self.sentiment_analyzer:
                return {'sentiment': 'neutral', 'confidence': 0.5, 'scores': {}}
            
            # Get sentiment scores
            results = self.sentiment_analyzer(text)
            
            # Process results
            sentiment_scores = {}
            max_score = 0
            dominant_sentiment = 'neutral'
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
                
                if score > max_score:
                    max_score = score
                    dominant_sentiment = label
            
            # Map labels to standard format
            sentiment_mapping = {
                'positive': 'bullish',
                'negative': 'bearish',
                'neutral': 'neutral',
                'bullish': 'bullish',
                'bearish': 'bearish'
            }
            
            mapped_sentiment = sentiment_mapping.get(dominant_sentiment, 'neutral')
            
            return {
                'sentiment': mapped_sentiment,
                'confidence': max_score,
                'scores': sentiment_scores,
                'raw_results': results
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'scores': {}}
    
    def recognize_patterns(self, data: pd.DataFrame):
        """Advanced pattern recognition using AI"""
        try:
            if data.empty or len(data) < 60:
                return {'patterns': [], 'confidence': 0.0}
            
            # Prepare data for pattern recognition
            features = self.prepare_pattern_features(data)
            
            if features is None:
                return {'patterns': [], 'confidence': 0.0}
            
            # Traditional pattern recognition
            traditional_patterns = self.detect_traditional_patterns(data)
            
            # AI-based pattern recognition
            ai_patterns = self.detect_ai_patterns(features)
            
            # Combine patterns
            all_patterns = traditional_patterns + ai_patterns
            
            # Calculate overall confidence
            confidence = np.mean([p['confidence'] for p in all_patterns]) if all_patterns else 0.0
            
            return {
                'patterns': all_patterns,
                'confidence': confidence,
                'pattern_count': len(all_patterns)
            }
            
        except Exception as e:
            print(f"Error recognizing patterns: {e}")
            return {'patterns': [], 'confidence': 0.0}
    
    def prepare_pattern_features(self, data: pd.DataFrame):
        """Prepare features for pattern recognition"""
        try:
            # Ensure we have OHLCV data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                return None
            
            # Create feature matrix
            features = []
            
            # Price features
            features.append(data['Close'].pct_change().fillna(0).values)
            features.append(data['High'].pct_change().fillna(0).values)
            features.append(data['Low'].pct_change().fillna(0).values)
            features.append(data['Volume'].pct_change().fillna(0).values)
            
            # Technical indicators as features
            sma_10 = data['Close'].rolling(window=10).mean()
            features.append((data['Close'] - sma_10).fillna(0).values)
            
            # Stack features
            feature_matrix = np.column_stack(features)
            
            return feature_matrix
            
        except Exception as e:
            print(f"Error preparing pattern features: {e}")
            return None
    
    def detect_traditional_patterns(self, data: pd.DataFrame):
        """Detect traditional chart patterns"""
        patterns = []
        
        try:
            # Double Top/Bottom patterns
            double_patterns = self.detect_double_patterns(data)
            patterns.extend(double_patterns)
            
            # Head and Shoulders
            h_s_patterns = self.detect_head_shoulders(data)
            patterns.extend(h_s_patterns)
            
            # Triangle patterns
            triangle_patterns = self.detect_triangles(data)
            patterns.extend(triangle_patterns)
            
            # Flag and Pennant patterns
            flag_patterns = self.detect_flags(data)
            patterns.extend(flag_patterns)
            
        except Exception as e:
            print(f"Error detecting traditional patterns: {e}")
        
        return patterns
    
    def detect_double_patterns(self, data: pd.DataFrame):
        """Detect double top/bottom patterns"""
        patterns = []
        
        try:
            if len(data) < 50:
                return patterns
            
            highs = data['High'].rolling(window=5).max()
            lows = data['Low'].rolling(window=5).min()
            
            # Simple double top detection
            recent_highs = highs.tail(20)
            max_high = recent_highs.max()
            
            # Look for two peaks of similar height
            peaks = recent_highs[recent_highs > max_high * 0.98]
            
            if len(peaks) >= 2:
                patterns.append({
                    'type': 'double_top',
                    'confidence': 0.7,
                    'signal': 'bearish',
                    'description': 'Double top pattern detected'
                })
            
            # Simple double bottom detection
            recent_lows = lows.tail(20)
            min_low = recent_lows.min()
            
            # Look for two troughs of similar depth
            troughs = recent_lows[recent_lows < min_low * 1.02]
            
            if len(troughs) >= 2:
                patterns.append({
                    'type': 'double_bottom',
                    'confidence': 0.7,
                    'signal': 'bullish',
                    'description': 'Double bottom pattern detected'
                })
            
        except Exception as e:
            print(f"Error detecting double patterns: {e}")
        
        return patterns
    
    def detect_head_shoulders(self, data: pd.DataFrame):
        """Detect head and shoulders patterns"""
        patterns = []
        
        try:
            if len(data) < 30:
                return patterns
            
            # Simplified head and shoulders detection
            close_prices = data['Close'].tail(30)
            
            # Find local maxima and minima
            from scipy.signal import argrelextrema
            
            maxima_idx = argrelextrema(close_prices.values, np.greater, order=3)[0]
            
            if len(maxima_idx) >= 3:
                # Check if middle peak is highest (head)
                if len(maxima_idx) >= 3:
                    left_shoulder = close_prices.iloc[maxima_idx[-3]]
                    head = close_prices.iloc[maxima_idx[-2]]
                    right_shoulder = close_prices.iloc[maxima_idx[-1]]
                    
                    if head > left_shoulder and head > right_shoulder:
                        # Check if shoulders are roughly equal
                        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                        
                        if shoulder_diff < 0.05:  # 5% tolerance
                            patterns.append({
                                'type': 'head_shoulders',
                                'confidence': 0.8,
                                'signal': 'bearish',
                                'description': 'Head and shoulders pattern detected'
                            })
            
        except Exception as e:
            print(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    def detect_triangles(self, data: pd.DataFrame):
        """Detect triangle patterns"""
        patterns = []
        
        try:
            if len(data) < 20:
                return patterns
            
            # Simplified triangle detection
            recent_data = data.tail(20)
            
            # Calculate trend lines
            highs = recent_data['High']
            lows = recent_data['Low']
            
            # Check for converging trend lines
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Ascending triangle
            if abs(high_trend) < 0.0001 and low_trend > 0:
                patterns.append({
                    'type': 'ascending_triangle',
                    'confidence': 0.6,
                    'signal': 'bullish',
                    'description': 'Ascending triangle pattern detected'
                })
            
            # Descending triangle
            elif abs(low_trend) < 0.0001 and high_trend < 0:
                patterns.append({
                    'type': 'descending_triangle',
                    'confidence': 0.6,
                    'signal': 'bearish',
                    'description': 'Descending triangle pattern detected'
                })
            
            # Symmetrical triangle
            elif high_trend < 0 and low_trend > 0:
                patterns.append({
                    'type': 'symmetrical_triangle',
                    'confidence': 0.5,
                    'signal': 'neutral',
                    'description': 'Symmetrical triangle pattern detected'
                })
            
        except Exception as e:
            print(f"Error detecting triangles: {e}")
        
        return patterns
    
    def detect_flags(self, data: pd.DataFrame):
        """Detect flag and pennant patterns"""
        patterns = []
        
        try:
            if len(data) < 15:
                return patterns
            
            # Simplified flag detection
            recent_data = data.tail(15)
            
            # Look for strong trend followed by consolidation
            first_half = recent_data.iloc[:7]
            second_half = recent_data.iloc[7:]
            
            # Calculate trend strength
            first_trend = (first_half['Close'].iloc[-1] - first_half['Close'].iloc[0]) / first_half['Close'].iloc[0]
            second_volatility = second_half['Close'].std() / second_half['Close'].mean()
            
            # Bull flag
            if first_trend > 0.02 and second_volatility < 0.01:
                patterns.append({
                    'type': 'bull_flag',
                    'confidence': 0.65,
                    'signal': 'bullish',
                    'description': 'Bull flag pattern detected'
                })
            
            # Bear flag
            elif first_trend < -0.02 and second_volatility < 0.01:
                patterns.append({
                    'type': 'bear_flag',
                    'confidence': 0.65,
                    'signal': 'bearish',
                    'description': 'Bear flag pattern detected'
                })
            
        except Exception as e:
            print(f"Error detecting flags: {e}")
        
        return patterns
    
    def detect_ai_patterns(self, features):
        """Detect patterns using AI models"""
        patterns = []
        
        try:
            if self.pattern_lstm is None or features is None:
                return patterns
            
            # Prepare data for LSTM
            if len(features) < 60:
                return patterns
            
            # Use the last 60 data points
            lstm_input = features[-60:].reshape(1, 60, -1)
            
            # Get prediction
            prediction = self.pattern_lstm.predict(lstm_input, verbose=0)
            
            # Interpret prediction
            class_names = ['bullish', 'bearish', 'neutral']
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            if confidence > 0.6:
                patterns.append({
                    'type': 'ai_pattern',
                    'confidence': confidence,
                    'signal': class_names[predicted_class],
                    'description': f'AI detected {class_names[predicted_class]} pattern'
                })
            
        except Exception as e:
            print(f"Error detecting AI patterns: {e}")
        
        return patterns
    
    def ensemble_prediction(self, features: pd.DataFrame):
        """Make ensemble prediction for trading signals"""
        try:
            if self.ensemble_model is None or features.empty:
                return {'prediction': 'hold', 'confidence': 0.5, 'probabilities': {}}
            
            # Prepare features
            X = self.prepare_ensemble_features(features)
            
            if X is None:
                return {'prediction': 'hold', 'confidence': 0.5, 'probabilities': {}}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Make prediction
            prediction = self.ensemble_model.predict(X_scaled)
            probabilities = self.ensemble_model.predict_proba(X_scaled)
            
            # Map prediction to action
            actions = ['buy', 'hold', 'sell']
            predicted_action = actions[prediction[0]]
            
            # Get confidence
            confidence = float(np.max(probabilities[0]))
            
            # Create probability dictionary
            prob_dict = {actions[i]: float(probabilities[0][i]) for i in range(len(actions))}
            
            return {
                'prediction': predicted_action,
                'confidence': confidence,
                'probabilities': prob_dict
            }
            
        except Exception as e:
            print(f"Error making ensemble prediction: {e}")
            return {'prediction': 'hold', 'confidence': 0.5, 'probabilities': {}}
    
    def prepare_ensemble_features(self, data: pd.DataFrame):
        """Prepare features for ensemble model"""
        try:
            if len(data) < 20:
                return None
            
            features = []
            
            # Price momentum features
            returns = data['Close'].pct_change().fillna(0)
            features.extend([
                returns.tail(5).mean(),
                returns.tail(10).mean(),
                returns.tail(20).mean(),
                returns.std()
            ])
            
            # Technical indicator features
            if 'RSI' in data.columns:
                features.append(data['RSI'].iloc[-1])
            else:
                features.append(50)  # Neutral RSI
            
            if 'MACD' in data.columns:
                features.append(data['MACD'].iloc[-1])
            else:
                features.append(0)
            
            if 'ADX' in data.columns:
                features.append(data['ADX'].iloc[-1])
            else:
                features.append(25)  # Neutral ADX
            
            # Volume features
            if 'Volume' in data.columns:
                volume_avg = data['Volume'].tail(20).mean()
                current_volume = data['Volume'].iloc[-1]
                features.append(current_volume / volume_avg if volume_avg > 0 else 1)
            else:
                features.append(1)
            
            # Volatility features
            volatility = returns.tail(20).std()
            features.append(volatility)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing ensemble features: {e}")
            return None
    
    def get_ai_insights(self, symbol: str, data: pd.DataFrame, news_text: str = ""):
        """Get comprehensive AI insights for a symbol"""
        try:
            insights = {
                'symbol': symbol,
                'sentiment': {'sentiment': 'neutral', 'confidence': 0.5},
                'patterns': {'patterns': [], 'confidence': 0.0},
                'ensemble': {'prediction': 'hold', 'confidence': 0.5},
                'ai_score': 0.5,
                'recommendations': []
            }
            
            # Analyze sentiment if news text is provided
            if news_text:
                insights['sentiment'] = self.analyze_sentiment(news_text)
            
            # Recognize patterns
            insights['patterns'] = self.recognize_patterns(data)
            
            # Ensemble prediction
            insights['ensemble'] = self.ensemble_prediction(data)
            
            # Calculate overall AI score
            sentiment_score = insights['sentiment']['confidence']
            pattern_score = insights['patterns']['confidence']
            ensemble_score = insights['ensemble']['confidence']
            
            insights['ai_score'] = (sentiment_score + pattern_score + ensemble_score) / 3
            
            # Generate recommendations
            insights['recommendations'] = self.generate_recommendations(insights)
            
            return insights
            
        except Exception as e:
            print(f"Error getting AI insights: {e}")
            return {
                'symbol': symbol,
                'sentiment': {'sentiment': 'neutral', 'confidence': 0.5},
                'patterns': {'patterns': [], 'confidence': 0.0},
                'ensemble': {'prediction': 'hold', 'confidence': 0.5},
                'ai_score': 0.5,
                'recommendations': []
            }
    
    def generate_recommendations(self, insights: dict):
        """Generate trading recommendations based on AI insights"""
        recommendations = []
        
        try:
            # Sentiment-based recommendations
            sentiment = insights['sentiment']['sentiment']
            sentiment_confidence = insights['sentiment']['confidence']
            
            if sentiment == 'bullish' and sentiment_confidence > 0.7:
                recommendations.append({
                    'type': 'sentiment',
                    'action': 'buy',
                    'reason': f'Strong bullish sentiment detected (confidence: {sentiment_confidence:.2f})',
                    'confidence': sentiment_confidence
                })
            elif sentiment == 'bearish' and sentiment_confidence > 0.7:
                recommendations.append({
                    'type': 'sentiment',
                    'action': 'sell',
                    'reason': f'Strong bearish sentiment detected (confidence: {sentiment_confidence:.2f})',
                    'confidence': sentiment_confidence
                })
            
            # Pattern-based recommendations
            patterns = insights['patterns']['patterns']
            for pattern in patterns:
                if pattern['confidence'] > 0.6:
                    action = 'buy' if pattern['signal'] == 'bullish' else 'sell' if pattern['signal'] == 'bearish' else 'hold'
                    recommendations.append({
                        'type': 'pattern',
                        'action': action,
                        'reason': f'{pattern["description"]} (confidence: {pattern["confidence"]:.2f})',
                        'confidence': pattern['confidence']
                    })
            
            # Ensemble-based recommendations
            ensemble_prediction = insights['ensemble']['prediction']
            ensemble_confidence = insights['ensemble']['confidence']
            
            if ensemble_confidence > 0.6:
                recommendations.append({
                    'type': 'ensemble',
                    'action': ensemble_prediction,
                    'reason': f'Ensemble model suggests {ensemble_prediction} (confidence: {ensemble_confidence:.2f})',
                    'confidence': ensemble_confidence
                })
            
            # Overall AI score recommendation
            ai_score = insights['ai_score']
            if ai_score > 0.7:
                recommendations.append({
                    'type': 'overall',
                    'action': 'strong_signal',
                    'reason': f'High AI confidence score: {ai_score:.2f}',
                    'confidence': ai_score
                })
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        
        return recommendations