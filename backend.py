"""
Backend API Server for Multi-Stock Options Analysis
File: backend_api.py
Deploy this on Render as a Web Service
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import requests
import time
from datetime import datetime
import pytz
import json
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
import os
from collections import deque
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Stock Options Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IST = pytz.timezone("Asia/Kolkata")

# Get from environment variables or use defaults
CLIENT_ID = os.getenv('DHAN_CLIENT_ID', '1100244268')
ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN', 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzYyODQzNzYzLCJpYXQiOjE3NjI3NTczNjMsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.6nJCQhdcbXkqA9YTJXSXXItAzDDhT9fuyJ7YuRWYSFV07gKfPrhJLYggGaCnz_8ww1LytC_DXRBZfz_pjvePLw')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7967747029:AAFyMl5zF1XvRqrhY5CIoR_1_EJwiEyrAqw')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '-470480347')

HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# In-memory storage
stock_data_cache = {}
stock_alerts_cache = {}
last_update_time = {}
stock_config = {}
rolling_data_cache = {}

class StockConfig(BaseModel):
    symbol: str
    scrip_id: int
    segment: str
    lot_size: int
    enabled: bool = True

class AnalysisResponse(BaseModel):
    symbol: str
    timestamp: str
    underlying_price: float
    data: Dict
    alerts: List[str]
    signals: Dict

# Telegram alert function
def send_telegram_alert(message: str):
    """Send alert to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, data=data, timeout=5)
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully")
        else:
            logger.warning(f"Telegram alert failed: {response.text}")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")

# Load stock configuration
def load_stock_config():
    """Load stock configuration from CSV file"""
    try:
        # Try to load from file
        if os.path.exists('stocks_config.csv'):
            df = pd.read_csv('stocks_config.csv')
        else:
            # Create default config
            default_data = {
                'symbol': ['NIFTY', 'BANKNIFTY', 'FINNIFTY'],
                'name': ['Nifty 50', 'Bank Nifty', 'Fin Nifty'],
                'scrip_id': [13, 25, 27],
                'segment': ['IDX_I', 'IDX_I', 'IDX_I'],
                'lot_size': [75, 25, 40],
                'enabled': [True, True, True]
            }
            df = pd.DataFrame(default_data)
            df.to_csv('stocks_config.csv', index=False)
            logger.info("Created default stocks_config.csv")
        
        config = {}
        for _, row in df.iterrows():
            if row.get('enabled', True):
                config[row['symbol']] = {
                    'scrip_id': int(row['scrip_id']),
                    'segment': row['segment'],
                    'lot_size': int(row['lot_size']),
                    'name': row.get('name', row['symbol'])
                }
        
        logger.info(f"Loaded {len(config)} stocks from config")
        return config
    except Exception as e:
        logger.error(f"Error loading stock config: {e}")
        # Return minimal default
        return {
            'NIFTY': {
                'scrip_id': 13,
                'segment': 'IDX_I',
                'lot_size': 75,
                'name': 'Nifty 50'
            }
        }

# Fetch expiry dates
def get_expiry_dates(scrip_id: int, segment: str):
    """Fetch expiry dates for a stock"""
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": scrip_id, "UnderlyingSeg": segment}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except Exception as e:
        logger.error(f"Error fetching expiry dates: {e}")
        return []

# Fetch option chain
def fetch_option_chain(scrip_id: int, segment: str, expiry: str):
    """Fetch option chain data"""
    url = "https://api.dhan.co/v2/optionchain"
    payload = {
        "UnderlyingScrip": scrip_id,
        "UnderlyingSeg": segment,
        "Expiry": expiry
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()
        time.sleep(2)  # Rate limiting
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return None

# Calculate sentiment score
def score_option_sentiment(row):
    """Calculate sentiment score for an option"""
    score = 0
    
    # OI change
    if row.get('OI_Change', 0) > 0:
        score += 1
    elif row.get('OI_Change', 0) < 0:
        score -= 1
    
    # LTP change
    if row.get('LTP_Change', 0) > 0:
        score += 1
    elif row.get('LTP_Change', 0) < 0:
        score -= 1
    
    # IV change
    if row.get('IV_Change', 0) > 0:
        score += 1
    elif row.get('IV_Change', 0) < 0:
        score -= 1
    
    # Theta
    if row.get('Theta', 0) < 0:
        score += 1
    elif row.get('Theta', 0) > 0:
        score -= 1
    
    # Vega
    if row.get('Vega', 0) > 0:
        score += 1
    elif row.get('Vega', 0) < 0:
        score -= 1
    
    # Bias
    if score >= 3:
        bias = "Aggressive Buying"
    elif score >= 1:
        bias = "Mild Buying"
    elif score == 0:
        bias = "Neutral"
    elif score <= -3:
        bias = "Aggressive Writing"
    else:
        bias = "Mild Writing"
    
    return score, bias

# Enhanced analysis function
def analyze_option_chain(option_chain, symbol: str):
    """Analyze option chain and generate comprehensive insights"""
    if not option_chain or "data" not in option_chain:
        return None
    
    try:
        option_chain_data = option_chain["data"]["oc"]
        underlying_price = option_chain["data"]["last_price"]
        
        data_list = []
        alerts = []
        
        # Initialize rolling data for this symbol if not exists
        if symbol not in rolling_data_cache:
            rolling_data_cache[symbol] = {}
        
        # Find ATM strike
        strikes = [float(k) for k in option_chain_data.keys()]
        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        # Determine strike interval
        if symbol in ['NIFTY', 'BANKNIFTY']:
            strike_interval = 50
        elif symbol == 'FINNIFTY':
            strike_interval = 50
        else:
            strike_interval = 100
        
        # Define range for ATM ± 5 strikes
        min_strike = atm_strike - 5 * strike_interval
        max_strike = atm_strike + 5 * strike_interval
        
        # Process strikes
        for strike, contracts in option_chain_data.items():
            strike_price = float(strike)
            
            if strike_price < min_strike or strike_price > max_strike:
                continue
            
            ce_data = contracts.get("ce", {})
            pe_data = contracts.get("pe", {})
            
            # Extract data for both CE and PE
            for opt_type, opt_data in [("CE", ce_data), ("PE", pe_data)]:
                if not opt_data:
                    continue
                
                iv = opt_data.get("implied_volatility", 0)
                oi = opt_data.get("oi", 0)
                ltp = opt_data.get("last_price", 0)
                volume = opt_data.get("volume", 0)
                greeks = opt_data.get("greeks", {})
                
                # Get previous data for changes
                key = f"{strike_price}_{opt_type}"
                prev_data = rolling_data_cache[symbol].get(key, {})
                
                # Calculate changes
                prev_iv = prev_data.get("IV", iv)
                prev_oi = prev_data.get("OI", oi)
                prev_ltp = prev_data.get("LTP", ltp)
                
                iv_change = ((iv - prev_iv) / prev_iv * 100) if prev_iv else 0
                oi_change = ((oi - prev_oi) / prev_oi * 100) if prev_oi else 0
                ltp_change = ((ltp - prev_ltp) / prev_ltp * 100) if prev_ltp else 0
                
                # Store current data
                rolling_data_cache[symbol][key] = {
                    "IV": iv,
                    "OI": oi,
                    "LTP": ltp
                }
                
                row_data = {
                    "StrikePrice": strike_price,
                    "Type": opt_type,
                    "IV": iv,
                    "OI": oi,
                    "LTP": ltp,
                    "Volume": volume,
                    "Delta": greeks.get("delta", 0),
                    "Gamma": greeks.get("gamma", 0),
                    "Theta": greeks.get("theta", 0),
                    "Vega": greeks.get("vega", 0),
                    "IV_Change": iv_change,
                    "OI_Change": oi_change,
                    "LTP_Change": ltp_change
                }
                
                # Calculate sentiment
                sentiment_score, sentiment_bias = score_option_sentiment(row_data)
                row_data["SentimentScore"] = sentiment_score
                row_data["SentimentBias"] = sentiment_bias
                
                data_list.append(row_data)
        
        df = pd.DataFrame(data_list)
        
        if df.empty:
            return None
        
        # Calculate metrics
        ce_df = df[df['Type'] == 'CE']
        pe_df = df[df['Type'] == 'PE']
        
        total_ce_oi = ce_df['OI'].sum()
        total_pe_oi = pe_df['OI'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Calculate weighted PCR
        ce_weighted = (ce_df['OI'] * ce_df['LTP']).sum()
        pe_weighted = (pe_df['OI'] * pe_df['LTP']).sum()
        pcr_weighted = pe_weighted / ce_weighted if ce_weighted > 0 else 0
        
        # Find max pain
        strike_oi = df.groupby('StrikePrice')['OI'].sum()
        max_pain = strike_oi.idxmax() if len(strike_oi) > 0 else atm_strike
        
        # Calculate IV metrics
        avg_ce_iv = ce_df['IV'].mean() if not ce_df.empty else 0
        avg_pe_iv = pe_df['IV'].mean() if not pe_df.empty else 0
        
        # Generate alerts
        if pcr > 1.5:
            alert = f"🔴 {symbol}: High PCR {pcr:.2f} - Strong Bearish sentiment"
            alerts.append(alert)
        elif pcr < 0.7:
            alert = f"🟢 {symbol}: Low PCR {pcr:.2f} - Strong Bullish sentiment"
            alerts.append(alert)
        
        # IV spike alerts
        high_iv_change = df[abs(df['IV_Change']) > 10]
        if not high_iv_change.empty:
            for _, row in high_iv_change.iterrows():
                alert = f"⚡ {symbol}: IV Spike at {row['StrikePrice']:.0f} {row['Type']} - {row['IV_Change']:.1f}%"
                alerts.append(alert)
        
        # OI surge alerts
        high_oi_change = df[df['OI_Change'] > 20]
        if not high_oi_change.empty:
            for _, row in high_oi_change.iterrows():
                alert = f"📊 {symbol}: OI Surge at {row['StrikePrice']:.0f} {row['Type']} - {row['OI_Change']:.1f}%"
                alerts.append(alert)
        
        # Sentiment alerts
        extreme_sentiment = df[abs(df['SentimentScore']) >= 3]
        if not extreme_sentiment.empty:
            for _, row in extreme_sentiment.iterrows():
                alert = f"🎯 {symbol}: {row['SentimentBias']} at {row['StrikePrice']:.0f} {row['Type']}"
                alerts.append(alert)
        
        # Distance from max pain
        distance_from_max_pain = abs(underlying_price - max_pain)
        max_pain_alert = ""
        if distance_from_max_pain > 100:
            direction = "above" if underlying_price > max_pain else "below"
            max_pain_alert = f"📍 {symbol}: Price {direction} Max Pain by {distance_from_max_pain:.0f} points"
            alerts.append(max_pain_alert)
        
        return {
            'df': df.to_dict('records'),
            'underlying_price': underlying_price,
            'atm_strike': atm_strike,
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi,
            'pcr': pcr,
            'pcr_weighted': pcr_weighted,
            'max_pain': max_pain,
            'avg_ce_iv': avg_ce_iv,
            'avg_pe_iv': avg_pe_iv,
            'alerts': alerts,
            'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            'strike_interval': strike_interval,
            'distance_from_max_pain': distance_from_max_pain
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# Background task to update all stocks
async def update_all_stocks():
    """Update data for all configured stocks"""
    logger.info("=" * 50)
    logger.info("Starting stock data update cycle")
    logger.info(f"Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    
    for symbol, config in stock_config.items():
        try:
            logger.info(f"Updating {symbol}...")
            
            # Get expiry dates
            expiry_dates = get_expiry_dates(config['scrip_id'], config['segment'])
            if not expiry_dates:
                logger.warning(f"No expiry dates found for {symbol}")
                continue
            
            nearest_expiry = expiry_dates[0]
            logger.info(f"{symbol} - Nearest expiry: {nearest_expiry}")
            
            # Fetch option chain
            option_chain = fetch_option_chain(
                config['scrip_id'],
                config['segment'],
                nearest_expiry
            )
            
            if option_chain:
                # Analyze
                analysis = analyze_option_chain(option_chain, symbol)
                
                if analysis:
                    stock_data_cache[symbol] = analysis
                    stock_alerts_cache[symbol] = analysis['alerts']
                    last_update_time[symbol] = datetime.now(IST)
                    
                    logger.info(f"✓ {symbol} updated - Price: {analysis['underlying_price']:.2f}, PCR: {analysis['pcr']:.2f}")
                    
                    # Send Telegram alerts for critical events
                    if analysis['alerts']:
                        top_alerts = analysis['alerts'][:3]  # Top 3 alerts
                        if top_alerts:
                            telegram_msg = f"<b>{symbol} Alerts</b>\n\n" + "\n".join(top_alerts)
                            send_telegram_alert(telegram_msg)
                else:
                    logger.warning(f"Analysis failed for {symbol}")
            else:
                logger.warning(f"Failed to fetch option chain for {symbol}")
            
            # Rate limiting between stocks
            await asyncio.sleep(3)
            
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
    
    logger.info("Stock data update cycle completed")
    logger.info("=" * 50)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global stock_config
    
    logger.info("🚀 Starting Multi-Stock Options Analysis API")
    logger.info(f"Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    stock_config = load_stock_config()
    logger.info(f"Loaded {len(stock_config)} stocks")
    
    # Initial data load
    logger.info("Performing initial data load...")
    await update_all_stocks()
    
    # Schedule periodic updates (every 5 minutes)
    scheduler = AsyncIOScheduler(timezone=IST)
    scheduler.add_job(update_all_stocks, 'interval', minutes=5)
    scheduler.start()
    logger.info("Scheduler started - Updates every 5 minutes")
    
    logger.info("✓ API is ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Multi-Stock Options Analysis API",
        "version": "1.0.0",
        "stocks_configured": len(stock_config),
        "stocks_cached": len(stock_data_cache),
        "last_update": max(last_update_time.values()).isoformat() if last_update_time else None,
        "time": datetime.now(IST).isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(IST).isoformat(),
        "stocks": {
            symbol: {
                "last_update": last_update_time.get(symbol, datetime.now(IST)).isoformat(),
                "has_data": symbol in stock_data_cache
            }
            for symbol in stock_config.keys()
        }
    }

@app.get("/api/stocks")
async def get_stocks():
    """Get list of all configured stocks"""
    stocks_list = []
    for symbol, config in stock_config.items():
        stocks_list.append({
            "symbol": symbol,
            "name": config.get('name', symbol),
            "last_update": last_update_time.get(symbol, datetime.now(IST)).isoformat(),
            "has_data": symbol in stock_data_cache,
            "alert_count": len(stock_alerts_cache.get(symbol, []))
        })
    
    return {"stocks": stocks_list, "total": len(stocks_list)}

@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get analysis data for a specific stock"""
    symbol = symbol.upper()
    
    if symbol not in stock_config:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not configured")
    
    if symbol not in stock_data_cache:
        raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    
    return {
        "symbol": symbol,
        "data": stock_data_cache[symbol],
        "last_update": last_update_time[symbol].isoformat()
    }

@app.get("/api/alerts")
async def get_all_alerts():
    """Get all alerts across stocks"""
    all_alerts = []
    for symbol, alerts in stock_alerts_cache.items():
        for alert in alerts:
            all_alerts.append({
                "symbol": symbol,
                "alert": alert,
                "timestamp": last_update_time.get(symbol, datetime.now(IST)).isoformat()
            })
    
    # Sort by timestamp (most recent first)
    all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        "alerts": all_alerts,
        "total": len(all_alerts)
    }

@app.get("/api/alerts/{symbol}")
async def get_stock_alerts(symbol: str):
    """Get alerts for a specific stock"""
    symbol = symbol.upper()
    
    if symbol not in stock_config:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not configured")
    
    if symbol not in stock_alerts_cache:
        return {
            "symbol": symbol,
            "alerts": [],
            "timestamp": datetime.now(IST).isoformat()
        }
    
    return {
        "symbol": symbol,
        "alerts": stock_alerts_cache[symbol],
        "timestamp": last_update_time.get(symbol, datetime.now(IST)).isoformat()
    }

@app.post("/api/refresh/{symbol}")
async def refresh_stock(symbol: str):
    """Manually refresh data for a specific stock"""
    symbol = symbol.upper()
    
    if symbol not in stock_config:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not configured")
    
    config = stock_config[symbol]
    
    try:
        logger.info(f"Manual refresh requested for {symbol}")
        
        expiry_dates = get_expiry_dates(config['scrip_id'], config['segment'])
        if not expiry_dates:
            raise HTTPException(status_code=500, detail="Failed to fetch expiry dates")
        
        option_chain = fetch_option_chain(
            config['scrip_id'],
            config['segment'],
            expiry_dates[0]
        )
        
        if not option_chain:
            raise HTTPException(status_code=500, detail="Failed to fetch option chain")
        
        analysis = analyze_option_chain(option_chain, symbol)
        
        if analysis:
            stock_data_cache[symbol] = analysis
            stock_alerts_cache[symbol] = analysis['alerts']
            last_update_time[symbol] = datetime.now(IST)
            
            logger.info(f"✓ {symbol} manually refreshed")
            
            return {
                "status": "success",
                "symbol": symbol,
                "timestamp": last_update_time[symbol].isoformat(),
                "data": analysis
            }
        else:
            raise HTTPException(status_code=500, detail="Analysis failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summary")
async def get_summary():
    """Get summary of all stocks"""
    summary = []
    
    for symbol in stock_config.keys():
        if symbol in stock_data_cache:
            data = stock_data_cache[symbol]
            summary.append({
                "symbol": symbol,
                "price": data['underlying_price'],
                "pcr": data['pcr'],
                "max_pain": data['max_pain'],
                "alert_count": len(data.get('alerts', [])),
                "last_update": last_update_time[symbol].isoformat()
            })
    
    return {
        "summary": summary,
        "timestamp": datetime.now(IST).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
