import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import logging
import warnings
import os
from dotenv import load_dotenv

# Import the Telegram bot manager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telegram import TelegramBotManager

# Load environment variables
load_dotenv()

warnings.filterwarnings("ignore")

# Set up logging for visibility in GitHub Actions/output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Telegram Bot Configuration
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID'))

# Initialize Telegram Bot
try:
    telegram_bot = TelegramBotManager(BOT_TOKEN, [CHAT_ID])
    TELEGRAM_ENABLED = telegram_bot.test_connection()
    if TELEGRAM_ENABLED:
        logging.info("✅ Telegram bot connected successfully")
    else:
        logging.warning("❌ Telegram bot connection failed")
        TELEGRAM_ENABLED = False
except Exception as e:
    logging.error(f"❌ Failed to initialize Telegram bot: {e}")
    TELEGRAM_ENABLED = False

def download_data(symbol, days=7):
    try:
        # Use 1-minute data for granular predictions
        # Limit to 7 days since 1-minute data is voluminous
        df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to download {symbol}: {e}")
        return pd.DataFrame()

def add_returns(df, name):
    # Calculate returns for 1-minute data
    df[f'{name}_Return'] = df['Close'].pct_change()
    
    # Predict next hour movement (60 minutes ahead) instead of next day
    df[f'{name}_Return_1h'] = df['Close'].pct_change(60)  # 1-hour return
    df[f'{name}_Target'] = df[f'{name}_Return_1h'].shift(-60).apply(lambda x: 1 if x > 0 else 0)
    
    # Reset index to handle datetime index for technical indicators
    series_index = df.index.values.ravel()
    series_close = df['Close'].reset_index(drop=True).values.ravel()
    named_series = pd.Series(series_close, index=series_index)
    
    # Adjust technical indicator periods for 1-minute data
    # RSI with 14-minute period (instead of 14 days)
    df[f'{name}_RSI'] = ta.momentum.RSIIndicator(close=named_series, window=14).rsi()
    
    # MACD with shorter periods for 1-minute data
    df[f'{name}_MACD'] = ta.trend.MACD(close=named_series, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    
    # Volatility over 30-minute rolling window (instead of 10 days)
    df[f'{name}_Volatility'] = df['Close'].rolling(30).std()
    
    return df

def scrape_market_sentiment():
    url = "https://news.google.com/search?q=nifty%20OR%20banknifty&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    sentiment_score = 0
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('a', class_='DY5T1d', limit=10)
        for h in headlines:
            text = h.text.lower()
            if any(word in text for word in ['rally', 'positive', 'gain', 'up', 'surge', 'buying']):
                sentiment_score += 1
            elif any(word in text for word in ['fall', 'crash', 'down', 'bearish', 'sell-off', 'decline']):
                sentiment_score -= 1
    except Exception as ex:
        logging.warning(f"Sentiment scrape failed: {ex}")
    return sentiment_score

def prepare_features():
    logging.info("⏬ Downloading 1-minute index data for last 7 days...")
    indices = {
        'NSEI': '^NSEI',             # Nifty 50
        'NSEBANK': '^NSEBANK',       # Banknifty
        'SNP': '^GSPC',              # S&P 500
        'INDIA_VIX': '^INDIAVIX',    # India VIX
        # Removed some indices that might not have reliable 1-minute data
        # 'N225': '^N225',             # Nikkei (different timezone)
        # 'HSI': '^HSI',               # Hang Seng (different timezone)
    }
    dfs = {}
    for name, symbol in indices.items():
        df = download_data(symbol, days=7)  # 7 days of 1-minute data
        if df.empty:
            logging.warning(f"No data for {name}, skipping...")
            continue
        df.reset_index(inplace=True)
        df = add_returns(df, name)
        dfs[name] = df
    
    if 'NSEI' not in dfs:
        raise ValueError("No data for NSEI (primary index)")
    
    # Create aligned dataframe based on NSEI timestamps
    df = pd.DataFrame(index=dfs['NSEI'].index)
    
    # Primary market data (NSEI and BANKNIFTY)
    df['N_Close'] = dfs['NSEI']['Close']
    df['N_Return'] = dfs['NSEI']['NSEI_Return']
    df['N_Return_1h'] = dfs['NSEI']['NSEI_Return_1h']
    
    if 'NSEBANK' in dfs:
        df['B_Close'] = dfs['NSEBANK']['Close']
        df['B_Return'] = dfs['NSEBANK']['NSEBANK_Return']
        df['B_Return_1h'] = dfs['NSEBANK']['NSEBANK_Return_1h']
    
    # International market influence (if available)
    if 'SNP' in dfs:
        df['SNP_Return'] = dfs['SNP']['SNP_Return']
    
    # VIX for volatility measure
    if 'INDIA_VIX' in dfs:
        df['VIX_Close'] = dfs['INDIA_VIX']['Close']
        df['VIX_Return'] = dfs['INDIA_VIX']['INDIA_VIX_Return']
    
    # Add technical indicators for primary indices
    for key in ['NSEI', 'NSEBANK']:
        if key in dfs:
            df[f'{key}_RSI'] = dfs[key][f'{key}_RSI']
            df[f'{key}_MACD'] = dfs[key][f'{key}_MACD']
            df[f'{key}_Volatility'] = dfs[key][f'{key}_Volatility']
    
    # Add time-based features for intraday patterns
    if not df.empty:
        # Assuming index is datetime
        timestamps = dfs['NSEI']['Datetime'] if 'Datetime' in dfs['NSEI'].columns else dfs['NSEI'].index
        df['Hour'] = pd.to_datetime(timestamps).dt.hour
        df['Minute'] = pd.to_datetime(timestamps).dt.minute
        df['DayOfWeek'] = pd.to_datetime(timestamps).dt.dayofweek
    
    # Simplified sentiment feature (single value for the session)
    df['Sentiment'] = scrape_market_sentiment()
    
    # Target variable
    df['Target'] = dfs['NSEI']['NSEI_Target']
    
    # Drop rows with NaN values (more aggressive cleaning for 1-minute data)
    df.dropna(inplace=True)
    
    logging.info(f"Prepared {len(df)} 1-minute data points for training")
    
    return df

def train_and_predict():
    logging.info("🔧 Preparing 1-minute data for hourly predictions...")
    df = prepare_features()
    if df.empty or len(df) < 500:  # Need more data points for 1-minute data
        raise ValueError("Not enough 1-minute data to train the model.")
    
    # Updated feature list for 1-minute data
    features = [
        'N_Return', 'N_Return_1h',
        'NSEI_RSI', 'NSEI_MACD', 'NSEI_Volatility',
        'Hour', 'Minute', 'DayOfWeek',  # Time-based features
        'Sentiment'
    ]
    
    # Add Banknifty features if available
    if 'B_Return' in df.columns:
        features.extend(['B_Return', 'B_Return_1h'])
    if 'NSEBANK_RSI' in df.columns:
        features.extend(['NSEBANK_RSI', 'NSEBANK_MACD', 'NSEBANK_Volatility'])
    
    # Add international market features if available
    if 'SNP_Return' in df.columns:
        features.append('SNP_Return')
    
    # Add VIX features if available
    if 'VIX_Return' in df.columns:
        features.extend(['VIX_Close', 'VIX_Return'])
    
    # Filter features that actually exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    logging.info(f"Using {len(features)} features: {features}")
    
    X = df[features]
    y = df['Target']
    
    # Time-aware split for training/validation (no shuffling)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X, y))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Random Forest classifier with adjusted parameters for more data
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, preds, output_dict=True)
    
    # Latest input prediction
    latest_input = X.iloc[[-1]]
    prob = model.predict_proba(latest_input)[0][1]
    trend = "📈 Uptrend" if prob > 0.6 else "📉 Downtrend" if prob < 0.4 else "➡️ Sideways"
    
    # Prepare results dictionary
    results = {
        'trend': trend,
        'probability': prob,
        'accuracy': report['accuracy'],
        'precision_0': report['0']['precision'],
        'precision_1': report['1']['precision'],
        'recall_0': report['0']['recall'],
        'recall_1': report['1']['recall'],
        'f1_0': report['0']['f1-score'],
        'f1_1': report['1']['f1-score'],
        'support_0': report['0']['support'],
        'support_1': report['1']['support'],
        'timestamp': datetime.now(),
        'data_points': len(df),
        'features_used': len(features)
    }
    
    # Print to console (keep existing behavior)
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, preds))
    print(f"\n📊 Predicted Market Trend for Next Hour: {trend}")
    print(f"➡️ Probability of Uptrend: {prob:.2f}")
    print(f"📈 Data points used: {len(df)}")
    print(f"🔧 Features used: {len(features)}")
    
    return results

def send_telegram_notification(results):
    """Send market prediction results via Telegram"""
    if not TELEGRAM_ENABLED:
        logging.warning("Telegram not enabled, skipping notification")
        return False
    
    try:
        # Format the prediction message
        trend = results['trend']
        prob = results['probability']
        accuracy = results['accuracy']
        timestamp = results['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        data_points = results.get('data_points', 'N/A')
        features_used = results.get('features_used', 'N/A')
        
        # Determine notification type based on trend
        if "Uptrend" in trend:
            notification_type = "success"
        elif "Downtrend" in trend:
            notification_type = "warning"
        else:
            notification_type = "info"
        
        # Create detailed message for hourly predictions
        message = f"""
<b>🤖 Hourly Market Prediction</b>
⏰ <i>{timestamp}</i>

<b>Next Hour Prediction:</b> {trend}
<b>Uptrend Probability:</b> {prob:.1%}
<b>Model Accuracy:</b> {accuracy:.1%}

<b>📊 Model Performance:</b>
• Precision (Down): {results['precision_0']:.2f}
• Precision (Up): {results['precision_1']:.2f}
• Recall (Down): {results['recall_0']:.2f}
• Recall (Up): {results['recall_1']:.2f}

<b>📈 Data Overview:</b>
• Data Points: {data_points} (1-min intervals)
• Features Used: {features_used}
• Timeframe: Last 7 days

<i>🔄(1-min data)</i>
        """.strip()
        
        # Send formatted notification
        success = telegram_bot.send_notification(message, parse_mode='HTML')
        
        if success:
            logging.info("✅ Telegram notification sent successfully")
        else:
            logging.error("❌ Failed to send Telegram notification")
        
        return success
        
    except Exception as e:
        logging.error(f"❌ Error sending Telegram notification: {e}")
        return False

def send_error_notification(error_msg):
    """Send error notification via Telegram"""
    if not TELEGRAM_ENABLED:
        return False
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = f"""
<b>🚨 Market Predictor Error</b>
⏰ <i>{timestamp}</i>

<b>Error:</b> {error_msg}

<i>Please check the logs for more details.</i>
        """.strip()
        
        return telegram_bot.send_formatted_notification(error_message, "error")
        
    except Exception as e:
        logging.error(f"Failed to send error notification: {e}")
        return False

if __name__ == "__main__":
    try:
        results = train_and_predict()
        send_telegram_notification(results)
    except Exception as e:
        logging.error(f"🚨 Pipeline failed: {e}", exc_info=True)
        send_error_notification(f"🚨 Pipeline failed: {e}")
