"""
News and Sentiment Analysis Service
Real-time news fetching and sentiment analysis for trading
"""

import requests
import feedparser
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import re
from textblob import TextBlob
import nltk
from newspaper import Article
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class NewsAndSentimentService:
    """Service for fetching and analyzing financial news"""
    
    def __init__(self):
        self.news_sources = {
            'forex_factory': 'https://www.forexfactory.com/news',
            'investing': 'https://www.investing.com/rss/news.rss',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'reuters': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114'
        }
        
        self.forex_keywords = [
            'forex', 'currency', 'exchange rate', 'dollar', 'euro', 'pound', 'yen',
            'usd', 'eur', 'gbp', 'jpy', 'cad', 'aud', 'chf', 'nzd',
            'federal reserve', 'ecb', 'bank of england', 'boj', 'central bank',
            'interest rate', 'monetary policy', 'inflation', 'gdp', 'employment',
            'trade war', 'brexit', 'economic data', 'nonfarm payrolls',
            'consumer price index', 'unemployment rate', 'retail sales'
        ]
        
        # Initialize sentiment analyzers
        self.init_sentiment_analyzers()
        
    def init_sentiment_analyzers(self):
        """Initialize sentiment analysis tools"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader_analyzer = None
            print("VADER sentiment analyzer not available")
    
    async def fetch_news_async(self, source_url: str, session: aiohttp.ClientSession) -> List[Dict]:
        """Fetch news from a source asynchronously"""
        try:
            async with session.get(source_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    return self.parse_rss_feed(content)
        except Exception as e:
            print(f"Error fetching news from {source_url}: {e}")
        return []
    
    def parse_rss_feed(self, xml_content: str) -> List[Dict]:
        """Parse RSS feed content"""
        try:
            feed = feedparser.parse(xml_content)
            news_items = []
            
            for entry in feed.entries[:10]:  # Limit to 10 items per source
                news_item = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed.feed.get('title', 'Unknown'),
                    'timestamp': datetime.now()
                }
                
                # Clean description
                if news_item['description']:
                    news_item['description'] = re.sub(r'<[^>]*>', '', news_item['description'])
                
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            print(f"Error parsing RSS feed: {e}")
            return []
    
    async def get_latest_news(self, max_items: int = 50) -> List[Dict]:
        """Get latest financial news from multiple sources"""
        all_news = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch news from all sources concurrently
                tasks = []
                for source_name, source_url in self.news_sources.items():
                    task = self.fetch_news_async(source_url, session)
                    tasks.append(task)
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, list):
                        all_news.extend(result)
                
                # Sort by timestamp and limit
                all_news.sort(key=lambda x: x['timestamp'], reverse=True)
                all_news = all_news[:max_items]
                
                # Filter for forex-related news
                forex_news = self.filter_forex_news(all_news)
                
                return forex_news
                
        except Exception as e:
            print(f"Error getting latest news: {e}")
            return []
    
    def filter_forex_news(self, news_items: List[Dict]) -> List[Dict]:
        """Filter news items for forex-related content"""
        forex_news = []
        
        for item in news_items:
            content = f"{item['title']} {item['description']}".lower()
            
            # Check if content contains forex keywords
            relevance_score = 0
            for keyword in self.forex_keywords:
                if keyword in content:
                    relevance_score += 1
            
            if relevance_score > 0:
                item['relevance_score'] = relevance_score
                forex_news.append(item)
        
        # Sort by relevance score
        forex_news.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return forex_news
    
    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze sentiment of news items"""
        if not news_items:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'confidence': 0.0
            }
        
        sentiments = []
        
        for item in news_items:
            text = f"{item['title']} {item['description']}"
            sentiment = self.analyze_text_sentiment(text)
            sentiment['news_item'] = item
            sentiments.append(sentiment)
        
        # Calculate overall sentiment
        overall_sentiment = self.calculate_overall_sentiment(sentiments)
        
        return overall_sentiment
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        sentiment_result = {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.0
        }
        
        try:
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # VADER sentiment analysis (if available)
            if self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                compound_score = vader_scores['compound']
                
                # Combine TextBlob and VADER scores
                combined_score = (polarity + compound_score) / 2
            else:
                combined_score = polarity
            
            # Determine sentiment
            if combined_score > 0.1:
                sentiment_result['sentiment'] = 'bullish'
            elif combined_score < -0.1:
                sentiment_result['sentiment'] = 'bearish'
            else:
                sentiment_result['sentiment'] = 'neutral'
            
            sentiment_result['score'] = combined_score
            sentiment_result['confidence'] = abs(combined_score)
            
        except Exception as e:
            print(f"Error analyzing text sentiment: {e}")
        
        return sentiment_result
    
    def calculate_overall_sentiment(self, sentiments: List[Dict]) -> Dict:
        """Calculate overall sentiment from multiple sentiment analyses"""
        if not sentiments:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'confidence': 0.0
            }
        
        bullish_count = sum(1 for s in sentiments if s['sentiment'] == 'bullish')
        bearish_count = sum(1 for s in sentiments if s['sentiment'] == 'bearish')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        
        total_count = len(sentiments)
        
        # Calculate weighted sentiment score
        total_score = sum(s['score'] for s in sentiments)
        avg_score = total_score / total_count if total_count > 0 else 0
        
        # Determine overall sentiment
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_sentiment = 'bullish'
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate confidence
        dominant_count = max(bullish_count, bearish_count, neutral_count)
        confidence = dominant_count / total_count if total_count > 0 else 0
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'confidence': confidence,
            'total_articles': total_count
        }
    
    def get_currency_specific_sentiment(self, currency: str, news_items: List[Dict]) -> Dict:
        """Get sentiment for a specific currency"""
        currency_keywords = {
            'USD': ['dollar', 'usd', 'federal reserve', 'fed', 'us economy'],
            'EUR': ['euro', 'eur', 'ecb', 'european central bank', 'eurozone'],
            'GBP': ['pound', 'gbp', 'bank of england', 'boe', 'uk economy', 'britain'],
            'JPY': ['yen', 'jpy', 'bank of japan', 'boj', 'japan economy'],
            'CAD': ['canadian dollar', 'cad', 'bank of canada', 'boc'],
            'AUD': ['australian dollar', 'aud', 'rba', 'reserve bank of australia'],
            'CHF': ['swiss franc', 'chf', 'swiss national bank', 'snb'],
            'NZD': ['new zealand dollar', 'nzd', 'rbnz', 'reserve bank of new zealand']
        }
        
        currency_news = []
        keywords = currency_keywords.get(currency.upper(), [currency.lower()])
        
        for item in news_items:
            content = f"{item['title']} {item['description']}".lower()
            
            # Check if content mentions the currency
            if any(keyword in content for keyword in keywords):
                currency_news.append(item)
        
        # Analyze sentiment for currency-specific news
        if currency_news:
            return self.analyze_news_sentiment(currency_news)
        else:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'confidence': 0.0
            }
    
    def get_market_sentiment_summary(self, news_items: List[Dict]) -> Dict:
        """Get comprehensive market sentiment summary"""
        try:
            # Overall market sentiment
            overall_sentiment = self.analyze_news_sentiment(news_items)
            
            # Currency-specific sentiments
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']
            currency_sentiments = {}
            
            for currency in major_currencies:
                currency_sentiments[currency] = self.get_currency_specific_sentiment(currency, news_items)
            
            # Market themes analysis
            themes = self.analyze_market_themes(news_items)
            
            # Risk sentiment
            risk_sentiment = self.analyze_risk_sentiment(news_items)
            
            return {
                'overall_sentiment': overall_sentiment,
                'currency_sentiments': currency_sentiments,
                'market_themes': themes,
                'risk_sentiment': risk_sentiment,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting market sentiment summary: {e}")
            return {
                'overall_sentiment': {'overall_sentiment': 'neutral', 'confidence': 0.0},
                'currency_sentiments': {},
                'market_themes': [],
                'risk_sentiment': 'neutral',
                'last_updated': datetime.now().isoformat()
            }
    
    def analyze_market_themes(self, news_items: List[Dict]) -> List[Dict]:
        """Analyze main market themes from news"""
        themes = []
        
        theme_keywords = {
            'inflation': ['inflation', 'cpi', 'consumer price', 'price pressure'],
            'interest_rates': ['interest rate', 'rate hike', 'rate cut', 'monetary policy'],
            'employment': ['employment', 'jobs', 'unemployment', 'nonfarm payrolls'],
            'trade': ['trade war', 'tariff', 'trade deal', 'export', 'import'],
            'geopolitics': ['geopolitical', 'war', 'sanctions', 'political'],
            'covid': ['covid', 'pandemic', 'lockdown', 'vaccine'],
            'energy': ['oil', 'energy', 'crude', 'gas', 'petroleum'],
            'recession': ['recession', 'gdp', 'economic growth', 'contraction']
        }
        
        for theme, keywords in theme_keywords.items():
            theme_count = 0
            theme_sentiment = []
            
            for item in news_items:
                content = f"{item['title']} {item['description']}".lower()
                
                if any(keyword in content for keyword in keywords):
                    theme_count += 1
                    sentiment = self.analyze_text_sentiment(content)
                    theme_sentiment.append(sentiment['score'])
            
            if theme_count > 0:
                avg_sentiment = sum(theme_sentiment) / len(theme_sentiment)
                themes.append({
                    'theme': theme,
                    'count': theme_count,
                    'sentiment_score': avg_sentiment,
                    'sentiment': 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral'
                })
        
        # Sort themes by count
        themes.sort(key=lambda x: x['count'], reverse=True)
        
        return themes[:5]  # Return top 5 themes
    
    def analyze_risk_sentiment(self, news_items: List[Dict]) -> str:
        """Analyze risk-on vs risk-off sentiment"""
        risk_on_keywords = ['risk-on', 'risk appetite', 'optimism', 'growth', 'recovery']
        risk_off_keywords = ['risk-off', 'safe haven', 'uncertainty', 'volatility', 'crisis']
        
        risk_on_count = 0
        risk_off_count = 0
        
        for item in news_items:
            content = f"{item['title']} {item['description']}".lower()
            
            if any(keyword in content for keyword in risk_on_keywords):
                risk_on_count += 1
            
            if any(keyword in content for keyword in risk_off_keywords):
                risk_off_count += 1
        
        if risk_on_count > risk_off_count:
            return 'risk-on'
        elif risk_off_count > risk_on_count:
            return 'risk-off'
        else:
            return 'neutral'
    
    async def get_real_time_sentiment(self) -> Dict:
        """Get real-time market sentiment"""
        try:
            # Fetch latest news
            news_items = await self.get_latest_news(max_items=100)
            
            # Analyze sentiment
            sentiment_summary = self.get_market_sentiment_summary(news_items)
            
            return sentiment_summary
            
        except Exception as e:
            print(f"Error getting real-time sentiment: {e}")
            return {
                'overall_sentiment': {'overall_sentiment': 'neutral', 'confidence': 0.0},
                'currency_sentiments': {},
                'market_themes': [],
                'risk_sentiment': 'neutral',
                'last_updated': datetime.now().isoformat()
            }

# Singleton instance
news_sentiment_service = NewsAndSentimentService()