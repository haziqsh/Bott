"""
Advanced Trading Strategies with State-of-the-Art Algorithms
Integrates: Volatility Clustering, Multi-timeframe Analysis, Statistical Arbitrage, Monte Carlo
"""

import numpy as np
import pandas as pd
from arch import arch_model
from hurst import compute_Hc
import stumpy
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedVolatilityStrategy:
    """
    Advanced volatility clustering strategy using GARCH models
    Based on Nobel Prize-winning volatility research
    """
    
    def __init__(self):
        self.name = "Advanced_Volatility_Clustering"
        self.garch_model = None
        self.volatility_regimes = {}
        self.hurst_exponent = None
        
    def calculate_garch_volatility(self, returns: pd.Series, p: int = 1, q: int = 1):
        """Calculate GARCH volatility with regime detection"""
        try:
            # Fit GARCH model
            garch_model = arch_model(returns, vol='GARCH', p=p, q=q)
            garch_fit = garch_model.fit(disp='off')
            
            # Get conditional volatility
            conditional_volatility = garch_fit.conditional_volatility
            
            # Detect volatility regimes
            volatility_regimes = self.detect_volatility_regimes(conditional_volatility)
            
            return {
                'conditional_volatility': conditional_volatility,
                'volatility_regimes': volatility_regimes,
                'garch_params': garch_fit.params,
                'aic': garch_fit.aic,
                'bic': garch_fit.bic
            }
            
        except Exception as e:
            print(f"Error calculating GARCH volatility: {e}")
            return None
    
    def detect_volatility_regimes(self, volatility: pd.Series):
        """Detect volatility regimes using clustering"""
        try:
            # Prepare data for clustering
            vol_data = volatility.values.reshape(-1, 1)
            
            # Use DBSCAN for regime detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            regimes = dbscan.fit_predict(vol_data)
            
            # Create regime dataframe
            regime_df = pd.DataFrame({
                'volatility': volatility,
                'regime': regimes,
                'timestamp': volatility.index
            })
            
            # Calculate regime statistics
            regime_stats = regime_df.groupby('regime').agg({
                'volatility': ['mean', 'std', 'min', 'max', 'count']
            }).round(6)
            
            return {
                'regime_labels': regimes,
                'regime_stats': regime_stats,
                'current_regime': regimes[-1] if len(regimes) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error detecting volatility regimes: {e}")
            return {'regime_labels': [], 'current_regime': 0}
    
    def calculate_hurst_exponent(self, price_series: pd.Series):
        """Calculate Hurst exponent for mean reversion/trending behavior"""
        try:
            if len(price_series) < 100:
                return 0.5
            
            # Calculate Hurst exponent
            H, c, data = compute_Hc(price_series.values, kind='price', simplified=True)
            
            # Interpret Hurst exponent
            if H < 0.5:
                behavior = "mean_reverting"
            elif H > 0.5:
                behavior = "trending"
            else:
                behavior = "random_walk"
            
            return {
                'hurst_exponent': H,
                'behavior': behavior,
                'confidence': abs(H - 0.5) * 2,
                'raw_data': data
            }
            
        except Exception as e:
            print(f"Error calculating Hurst exponent: {e}")
            return {'hurst_exponent': 0.5, 'behavior': 'random_walk', 'confidence': 0.0}
    
    def generate_volatility_signals(self, data: pd.DataFrame):
        """Generate trading signals based on volatility analysis"""
        try:
            if len(data) < 100:
                return []
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate GARCH volatility
            garch_results = self.calculate_garch_volatility(returns)
            if not garch_results:
                return []
            
            # Calculate Hurst exponent
            hurst_results = self.calculate_hurst_exponent(data['close'])
            
            # Generate signals based on volatility regime and market behavior
            signals = []
            
            current_regime = garch_results['volatility_regimes']['current_regime']
            current_vol = garch_results['conditional_volatility'].iloc[-1]
            hurst_exp = hurst_results['hurst_exponent']
            
            # Low volatility regime + trending behavior = momentum signal
            if current_regime == 0 and hurst_exp > 0.6:  # Low vol + trending
                price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
                
                if price_momentum > 0.01:
                    signals.append({
                        'type': 'BUY',
                        'strategy': 'Volatility_Momentum',
                        'strength': min(0.9, 0.7 + hurst_results['confidence']),
                        'reason': f'Low volatility regime + trending behavior (H={hurst_exp:.3f})',
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * 0.995,
                        'take_profit': data['close'].iloc[-1] * 1.015,
                        'volatility_regime': current_regime,
                        'hurst_exponent': hurst_exp
                    })
                elif price_momentum < -0.01:
                    signals.append({
                        'type': 'SELL',
                        'strategy': 'Volatility_Momentum',
                        'strength': min(0.9, 0.7 + hurst_results['confidence']),
                        'reason': f'Low volatility regime + trending behavior (H={hurst_exp:.3f})',
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * 1.005,
                        'take_profit': data['close'].iloc[-1] * 0.985,
                        'volatility_regime': current_regime,
                        'hurst_exponent': hurst_exp
                    })
            
            # High volatility regime + mean reversion = contrarian signal
            elif current_regime > 0 and hurst_exp < 0.4:  # High vol + mean reversion
                # Look for extreme price movements to fade
                price_zscore = (data['close'].iloc[-1] - data['close'].rolling(20).mean().iloc[-1]) / data['close'].rolling(20).std().iloc[-1]
                
                if price_zscore > 2:  # Overbought
                    signals.append({
                        'type': 'SELL',
                        'strategy': 'Volatility_Mean_Reversion',
                        'strength': min(0.85, 0.6 + hurst_results['confidence']),
                        'reason': f'High volatility regime + mean reversion (H={hurst_exp:.3f}, Z={price_zscore:.2f})',
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * 1.01,
                        'take_profit': data['close'].iloc[-1] * 0.99,
                        'volatility_regime': current_regime,
                        'hurst_exponent': hurst_exp
                    })
                elif price_zscore < -2:  # Oversold
                    signals.append({
                        'type': 'BUY',
                        'strategy': 'Volatility_Mean_Reversion',
                        'strength': min(0.85, 0.6 + hurst_results['confidence']),
                        'reason': f'High volatility regime + mean reversion (H={hurst_exp:.3f}, Z={price_zscore:.2f})',
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * 0.99,
                        'take_profit': data['close'].iloc[-1] * 1.01,
                        'volatility_regime': current_regime,
                        'hurst_exponent': hurst_exp
                    })
            
            return signals
            
        except Exception as e:
            print(f"Error generating volatility signals: {e}")
            return []

class MultiTimeframeCorrelationStrategy:
    """
    Advanced multi-timeframe correlation analysis
    Detects cross-timeframe patterns and correlations
    """
    
    def __init__(self):
        self.name = "Multi_Timeframe_Correlation"
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.correlation_matrix = None
        self.regime_detector = None
        
    def calculate_multi_timeframe_features(self, data: pd.DataFrame):
        """Calculate features across multiple timeframes"""
        try:
            features = {}
            
            # Short-term features (1-5 periods)
            features['short_momentum'] = data['close'].pct_change(5).fillna(0)
            features['short_volatility'] = data['close'].pct_change().rolling(5).std().fillna(0)
            features['short_rsi'] = self.calculate_rsi(data['close'], 5)
            
            # Medium-term features (10-20 periods)
            features['medium_momentum'] = data['close'].pct_change(20).fillna(0)
            features['medium_volatility'] = data['close'].pct_change().rolling(20).std().fillna(0)
            features['medium_rsi'] = self.calculate_rsi(data['close'], 14)
            
            # Long-term features (50-100 periods)
            features['long_momentum'] = data['close'].pct_change(50).fillna(0)
            features['long_volatility'] = data['close'].pct_change().rolling(50).std().fillna(0)
            features['long_rsi'] = self.calculate_rsi(data['close'], 21)
            
            # Cross-timeframe ratios
            features['momentum_ratio'] = features['short_momentum'] / (features['medium_momentum'] + 1e-6)
            features['volatility_ratio'] = features['short_volatility'] / (features['medium_volatility'] + 1e-6)
            features['rsi_divergence'] = features['short_rsi'] - features['medium_rsi']
            
            return pd.DataFrame(features)
            
        except Exception as e:
            print(f"Error calculating multi-timeframe features: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def detect_cross_timeframe_patterns(self, features: pd.DataFrame):
        """Detect patterns across different timeframes"""
        try:
            if features.empty or len(features) < 50:
                return []
            
            patterns = []
            
            # Get latest values
            latest = features.iloc[-1]
            
            # Pattern 1: Momentum Alignment
            if (latest['short_momentum'] > 0 and 
                latest['medium_momentum'] > 0 and 
                latest['long_momentum'] > 0):
                patterns.append({
                    'pattern': 'bullish_momentum_alignment',
                    'strength': 0.8,
                    'timeframes': ['short', 'medium', 'long'],
                    'signal': 'bullish'
                })
            elif (latest['short_momentum'] < 0 and 
                  latest['medium_momentum'] < 0 and 
                  latest['long_momentum'] < 0):
                patterns.append({
                    'pattern': 'bearish_momentum_alignment',
                    'strength': 0.8,
                    'timeframes': ['short', 'medium', 'long'],
                    'signal': 'bearish'
                })
            
            # Pattern 2: Volatility Compression
            if (latest['volatility_ratio'] < 0.5 and 
                latest['short_volatility'] < features['short_volatility'].rolling(20).mean().iloc[-1]):
                patterns.append({
                    'pattern': 'volatility_compression',
                    'strength': 0.7,
                    'timeframes': ['short', 'medium'],
                    'signal': 'breakout_pending'
                })
            
            # Pattern 3: RSI Divergence
            if abs(latest['rsi_divergence']) > 20:
                signal = 'bearish' if latest['rsi_divergence'] > 0 else 'bullish'
                patterns.append({
                    'pattern': 'rsi_divergence',
                    'strength': 0.6,
                    'timeframes': ['short', 'medium'],
                    'signal': signal
                })
            
            # Pattern 4: Momentum Reversal
            if (latest['momentum_ratio'] > 2 and latest['short_momentum'] > 0.02):
                patterns.append({
                    'pattern': 'momentum_exhaustion',
                    'strength': 0.65,
                    'timeframes': ['short', 'medium'],
                    'signal': 'bearish'
                })
            elif (latest['momentum_ratio'] < -2 and latest['short_momentum'] < -0.02):
                patterns.append({
                    'pattern': 'momentum_exhaustion',
                    'strength': 0.65,
                    'timeframes': ['short', 'medium'],
                    'signal': 'bullish'
                })
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting cross-timeframe patterns: {e}")
            return []
    
    def generate_correlation_signals(self, data: pd.DataFrame):
        """Generate trading signals based on multi-timeframe correlation"""
        try:
            if len(data) < 100:
                return []
            
            # Calculate multi-timeframe features
            features = self.calculate_multi_timeframe_features(data)
            
            if features.empty:
                return []
            
            # Detect cross-timeframe patterns
            patterns = self.detect_cross_timeframe_patterns(features)
            
            signals = []
            
            for pattern in patterns:
                if pattern['signal'] in ['bullish', 'bearish'] and pattern['strength'] > 0.6:
                    signal_type = 'BUY' if pattern['signal'] == 'bullish' else 'SELL'
                    
                    # Calculate entry and exit prices
                    current_price = data['close'].iloc[-1]
                    atr = self.calculate_atr(data)
                    
                    if signal_type == 'BUY':
                        stop_loss = current_price - (2 * atr)
                        take_profit = current_price + (3 * atr)
                    else:
                        stop_loss = current_price + (2 * atr)
                        take_profit = current_price - (3 * atr)
                    
                    signals.append({
                        'type': signal_type,
                        'strategy': 'Multi_Timeframe_Correlation',
                        'strength': pattern['strength'],
                        'reason': f"{pattern['pattern']} across {', '.join(pattern['timeframes'])} timeframes",
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pattern': pattern['pattern'],
                        'timeframes': pattern['timeframes']
                    })
            
            return signals
            
        except Exception as e:
            print(f"Error generating correlation signals: {e}")
            return []
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14):
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            return true_range.rolling(period).mean().iloc[-1]
        except:
            return data['close'].iloc[-1] * 0.01  # 1% fallback

class AdvancedStatisticalArbitrageStrategy:
    """
    Advanced statistical arbitrage using cointegration and mean reversion
    Detects statistical relationships between instruments
    """
    
    def __init__(self):
        self.name = "Advanced_Statistical_Arbitrage"
        self.cointegration_pairs = []
        self.mean_reversion_models = {}
        self.anomaly_detector = IsolationForest(contamination=0.1)
        
    def calculate_cointegration_score(self, series1: pd.Series, series2: pd.Series):
        """Calculate cointegration score between two series"""
        try:
            from statsmodels.tsa.stattools import coint
            
            # Perform cointegration test
            coint_t, p_value, crit_values = coint(series1, series2)
            
            # Calculate correlation
            correlation = series1.corr(series2)
            
            # Calculate spread
            spread = series1 - series2
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Calculate half-life of mean reversion
            half_life = self.calculate_half_life(spread)
            
            return {
                'cointegration_score': coint_t,
                'p_value': p_value,
                'correlation': correlation,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'half_life': half_life,
                'is_cointegrated': p_value < 0.05
            }
            
        except Exception as e:
            print(f"Error calculating cointegration: {e}")
            return None
    
    def calculate_half_life(self, spread: pd.Series):
        """Calculate half-life of mean reversion"""
        try:
            # Use AR(1) model to estimate half-life
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()
            
            # Remove NaN values
            valid_idx = ~(spread_lag.isna() | spread_diff.isna())
            spread_lag = spread_lag[valid_idx]
            spread_diff = spread_diff[valid_idx]
            
            if len(spread_lag) < 10:
                return float('inf')
            
            # Fit AR(1) model
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
            
            # Calculate half-life
            lambda_param = reg.coef_[0]
            
            if lambda_param >= 0:
                return float('inf')
            
            half_life = -np.log(2) / lambda_param
            
            return half_life if half_life > 0 else float('inf')
            
        except Exception as e:
            print(f"Error calculating half-life: {e}")
            return float('inf')
    
    def detect_statistical_anomalies(self, data: pd.DataFrame):
        """Detect statistical anomalies using isolation forest"""
        try:
            if len(data) < 50:
                return []
            
            # Prepare features for anomaly detection
            features = []
            
            # Price features
            returns = data['close'].pct_change().fillna(0)
            features.append(returns.values)
            
            # Volatility features
            volatility = returns.rolling(20).std().fillna(0)
            features.append(volatility.values)
            
            # Volume features (if available)
            if 'volume' in data.columns:
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
                features.append(volume_ratio.fillna(1).values)
            else:
                features.append(np.ones(len(data)))
            
            # Combine features
            feature_matrix = np.column_stack(features)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.fit_predict(feature_matrix)
            
            # Get anomaly scores
            anomaly_scores = self.anomaly_detector.decision_function(feature_matrix)
            
            # Find recent anomalies
            recent_anomalies = []
            for i in range(len(anomalies) - 20, len(anomalies)):  # Check last 20 periods
                if i >= 0 and anomalies[i] == -1:  # Anomaly detected
                    recent_anomalies.append({
                        'index': i,
                        'anomaly_score': anomaly_scores[i],
                        'price': data['close'].iloc[i],
                        'return': returns.iloc[i],
                        'volatility': volatility.iloc[i]
                    })
            
            return recent_anomalies
            
        except Exception as e:
            print(f"Error detecting statistical anomalies: {e}")
            return []
    
    def generate_arbitrage_signals(self, data: pd.DataFrame):
        """Generate statistical arbitrage signals"""
        try:
            if len(data) < 100:
                return []
            
            signals = []
            
            # Calculate z-score for mean reversion
            prices = data['close']
            rolling_mean = prices.rolling(window=20).mean()
            rolling_std = prices.rolling(window=20).std()
            z_score = (prices - rolling_mean) / rolling_std
            
            current_z_score = z_score.iloc[-1]
            
            # Detect statistical anomalies
            anomalies = self.detect_statistical_anomalies(data)
            
            # Generate signals based on z-score extremes
            if current_z_score > 2:  # Price too high
                signals.append({
                    'type': 'SELL',
                    'strategy': 'Statistical_Arbitrage',
                    'strength': min(0.9, 0.5 + abs(current_z_score) * 0.1),
                    'reason': f'Statistical mean reversion - price {current_z_score:.2f} std devs above mean',
                    'entry_price': data['close'].iloc[-1],
                    'stop_loss': data['close'].iloc[-1] * 1.01,
                    'take_profit': rolling_mean.iloc[-1],
                    'z_score': current_z_score,
                    'expected_return_time': self.calculate_half_life(z_score)
                })
            elif current_z_score < -2:  # Price too low
                signals.append({
                    'type': 'BUY',
                    'strategy': 'Statistical_Arbitrage',
                    'strength': min(0.9, 0.5 + abs(current_z_score) * 0.1),
                    'reason': f'Statistical mean reversion - price {current_z_score:.2f} std devs below mean',
                    'entry_price': data['close'].iloc[-1],
                    'stop_loss': data['close'].iloc[-1] * 0.99,
                    'take_profit': rolling_mean.iloc[-1],
                    'z_score': current_z_score,
                    'expected_return_time': self.calculate_half_life(z_score)
                })
            
            # Generate signals based on anomalies
            for anomaly in anomalies[-5:]:  # Check last 5 anomalies
                if anomaly['anomaly_score'] < -0.5:  # Strong anomaly
                    signal_type = 'SELL' if anomaly['return'] > 0 else 'BUY'
                    
                    signals.append({
                        'type': signal_type,
                        'strategy': 'Statistical_Anomaly',
                        'strength': min(0.8, 0.5 + abs(anomaly['anomaly_score']) * 0.3),
                        'reason': f'Statistical anomaly detected - score: {anomaly["anomaly_score"]:.3f}',
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * (1.005 if signal_type == 'SELL' else 0.995),
                        'take_profit': data['close'].iloc[-1] * (0.995 if signal_type == 'SELL' else 1.005),
                        'anomaly_score': anomaly['anomaly_score']
                    })
            
            return signals
            
        except Exception as e:
            print(f"Error generating arbitrage signals: {e}")
            return []

class MonteCarloRiskAnalysis:
    """
    Monte Carlo simulations for risk analysis and strategy optimization
    """
    
    def __init__(self):
        self.name = "Monte_Carlo_Risk_Analysis"
        self.simulations = 1000
        self.confidence_levels = [0.95, 0.99]
        
    def simulate_price_paths(self, current_price: float, volatility: float, drift: float, days: int = 30):
        """Simulate multiple price paths using Monte Carlo"""
        try:
            dt = 1/252  # Daily time step
            paths = []
            
            for _ in range(self.simulations):
                price_path = [current_price]
                
                for _ in range(days):
                    # Geometric Brownian Motion
                    random_shock = np.random.normal(0, 1)
                    price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
                    new_price = price_path[-1] * np.exp(price_change)
                    price_path.append(new_price)
                
                paths.append(price_path)
            
            return np.array(paths)
            
        except Exception as e:
            print(f"Error simulating price paths: {e}")
            return np.array([])
    
    def calculate_var_cvar(self, price_paths: np.ndarray, initial_investment: float = 10000):
        """Calculate Value at Risk (VaR) and Conditional VaR"""
        try:
            if price_paths.size == 0:
                return {}
            
            # Calculate returns for each path
            final_prices = price_paths[:, -1]
            returns = (final_prices - price_paths[:, 0]) / price_paths[:, 0]
            portfolio_values = initial_investment * (1 + returns)
            portfolio_returns = returns
            
            # Calculate VaR and CVaR for different confidence levels
            var_cvar_results = {}
            
            for conf_level in self.confidence_levels:
                # VaR (Value at Risk)
                var_percentile = (1 - conf_level) * 100
                var_return = np.percentile(portfolio_returns, var_percentile)
                var_dollar = initial_investment * var_return
                
                # CVaR (Conditional Value at Risk)
                cvar_returns = portfolio_returns[portfolio_returns <= var_return]
                cvar_return = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_return
                cvar_dollar = initial_investment * cvar_return
                
                var_cvar_results[f'{conf_level*100:.0f}%'] = {
                    'var_return': var_return,
                    'var_dollar': var_dollar,
                    'cvar_return': cvar_return,
                    'cvar_dollar': cvar_dollar
                }
            
            # Additional risk metrics
            expected_return = np.mean(portfolio_returns)
            volatility = np.std(portfolio_returns)
            sharpe_ratio = expected_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns.reshape(-1, 1), axis=0)
            running_max = np.maximum.accumulate(cumulative_returns, axis=0)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            return {
                'var_cvar': var_cvar_results,
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'probability_of_loss': np.mean(portfolio_returns < 0),
                'probability_of_gain': np.mean(portfolio_returns > 0)
            }
            
        except Exception as e:
            print(f"Error calculating VaR/CVaR: {e}")
            return {}
    
    def optimize_position_size(self, expected_return: float, volatility: float, risk_tolerance: float = 0.02):
        """Optimize position size using Kelly criterion and Monte Carlo"""
        try:
            # Kelly criterion for optimal position sizing
            if volatility == 0:
                return 0
            
            # Simplified Kelly formula
            kelly_fraction = expected_return / (volatility ** 2)
            
            # Apply risk tolerance constraint
            max_position = risk_tolerance
            optimal_position = min(abs(kelly_fraction), max_position)
            
            # Monte Carlo validation
            position_sizes = np.linspace(0.01, 0.05, 50)
            sharpe_ratios = []
            
            for pos_size in position_sizes:
                simulated_returns = []
                
                for _ in range(100):  # Quick simulation
                    random_return = np.random.normal(expected_return, volatility)
                    portfolio_return = pos_size * random_return
                    simulated_returns.append(portfolio_return)
                
                avg_return = np.mean(simulated_returns)
                return_volatility = np.std(simulated_returns)
                sharpe = avg_return / return_volatility if return_volatility > 0 else 0
                sharpe_ratios.append(sharpe)
            
            # Find optimal position size
            optimal_idx = np.argmax(sharpe_ratios)
            monte_carlo_optimal = position_sizes[optimal_idx]
            
            return {
                'kelly_fraction': kelly_fraction,
                'risk_adjusted_position': optimal_position,
                'monte_carlo_optimal': monte_carlo_optimal,
                'expected_sharpe': sharpe_ratios[optimal_idx]
            }
            
        except Exception as e:
            print(f"Error optimizing position size: {e}")
            return {'kelly_fraction': 0, 'risk_adjusted_position': 0.01}
    
    def generate_risk_assessment(self, data: pd.DataFrame):
        """Generate comprehensive risk assessment"""
        try:
            if len(data) < 30:
                return {}
            
            # Calculate historical metrics
            returns = data['close'].pct_change().dropna()
            current_price = data['close'].iloc[-1]
            historical_volatility = returns.std() * np.sqrt(252)  # Annualized
            historical_drift = returns.mean() * 252  # Annualized
            
            # Run Monte Carlo simulation
            price_paths = self.simulate_price_paths(
                current_price, 
                historical_volatility, 
                historical_drift
            )
            
            if price_paths.size == 0:
                return {}
            
            # Calculate risk metrics
            risk_metrics = self.calculate_var_cvar(price_paths)
            
            # Optimize position sizing
            position_optimization = self.optimize_position_size(
                historical_drift, 
                historical_volatility
            )
            
            # Generate risk signals
            risk_signals = []
            
            # High volatility warning
            if historical_volatility > 0.3:  # 30% annualized volatility
                risk_signals.append({
                    'type': 'WARNING',
                    'category': 'High_Volatility',
                    'message': f'High volatility detected: {historical_volatility:.1%} annualized',
                    'recommended_action': 'Reduce position size or increase stop-loss'
                })
            
            # Negative expected return warning
            if historical_drift < -0.1:  # -10% annualized drift
                risk_signals.append({
                    'type': 'WARNING',
                    'category': 'Negative_Drift',
                    'message': f'Negative expected return: {historical_drift:.1%} annualized',
                    'recommended_action': 'Consider avoiding long positions'
                })
            
            # High drawdown risk
            if risk_metrics.get('max_drawdown', 0) < -0.2:  # 20% max drawdown
                risk_signals.append({
                    'type': 'CAUTION',
                    'category': 'High_Drawdown_Risk',
                    'message': f'High drawdown risk: {risk_metrics["max_drawdown"]:.1%}',
                    'recommended_action': 'Implement strict stop-loss rules'
                })
            
            return {
                'historical_metrics': {
                    'volatility': historical_volatility,
                    'drift': historical_drift,
                    'current_price': current_price
                },
                'risk_metrics': risk_metrics,
                'position_optimization': position_optimization,
                'risk_signals': risk_signals,
                'monte_carlo_paths': price_paths[-5:].tolist()  # Last 5 paths for visualization
            }
            
        except Exception as e:
            print(f"Error generating risk assessment: {e}")
            return {}

# Factory class to manage all advanced strategies
class AdvancedStrategyFactory:
    """Factory class to manage all advanced strategies"""
    
    def __init__(self):
        self.strategies = {
            'volatility': AdvancedVolatilityStrategy(),
            'correlation': MultiTimeframeCorrelationStrategy(),
            'arbitrage': AdvancedStatisticalArbitrageStrategy(),
            'risk_analysis': MonteCarloRiskAnalysis()
        }
    
    def get_all_signals(self, data: pd.DataFrame, symbol: str = ""):
        """Get signals from all advanced strategies"""
        all_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy_name == 'volatility':
                    signals = strategy.generate_volatility_signals(data)
                elif strategy_name == 'correlation':
                    signals = strategy.generate_correlation_signals(data)
                elif strategy_name == 'arbitrage':
                    signals = strategy.generate_arbitrage_signals(data)
                else:
                    continue  # Skip risk analysis for signal generation
                
                # Add metadata to signals
                for signal in signals:
                    signal['symbol'] = symbol
                    signal['timestamp'] = pd.Timestamp.now().isoformat()
                    signal['strategy_category'] = 'advanced'
                
                all_signals.extend(signals)
                
            except Exception as e:
                print(f"Error getting signals from {strategy_name}: {e}")
        
        return all_signals
    
    def get_risk_assessment(self, data: pd.DataFrame, symbol: str = ""):
        """Get comprehensive risk assessment"""
        try:
            risk_analysis = self.strategies['risk_analysis'].generate_risk_assessment(data)
            risk_analysis['symbol'] = symbol
            risk_analysis['timestamp'] = pd.Timestamp.now().isoformat()
            return risk_analysis
        except Exception as e:
            print(f"Error getting risk assessment: {e}")
            return {}