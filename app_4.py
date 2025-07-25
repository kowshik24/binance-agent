import streamlit as st
from openai import OpenAI
import instructor
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from tavily import TavilyClient
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
from pycoingecko import CoinGeckoAPI
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import base64
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# --- Initial Setup & Configuration ---
load_dotenv()
st.set_page_config(
    page_title="AI Investment Committee Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

### NEW: Centralized configuration for easier changes ###
CONFIG = {
    "kline_interval_ta": Client.KLINE_INTERVAL_4HOUR,
    "kline_history_ta": "3 months ago UTC",
    "kline_interval_perf": Client.KLINE_INTERVAL_1DAY,
    "kline_history_perf": "8 day ago UTC",
    "risk_profiles": ["Conservative", "Balanced", "Aggressive"],
    "quote_currencies": ["USDT", "BUSD", "USDC", "FDUSD", "BTC", "ETH"], # Order of preference
    "min_usd_value_display": 1.0,
    "llm_model": "gpt-4o",
    "top_cryptos_count": 50,  # Reduced from 100 to avoid API limits
    "portfolio_health_weights": {
        "diversification": 0.25,
        "performance": 0.25,
        "risk_management": 0.25,
        "sentiment": 0.25
    }
}

# --- Pydantic Models ---
class ActionableTrade(BaseModel):
    # ... (no changes needed to your excellent Pydantic models) ...
    action: Literal["BUY", "SELL", "HOLD"] = Field(..., description="The recommended action.")
    coin: str = Field(..., description="The cryptocurrency ticker symbol (e.g., 'BTC').")
    amount_usd: Optional[float] = Field(None, description="Suggested USD amount for the transaction.")
    reasoning: str = Field(..., description="Clear, concise reason for the recommended action, referencing news or technicals.")
    entry_price: Optional[float] = Field(None, description="Suggested entry price for a BUY order.")
    take_profit_price: Optional[float] = Field(None, description="Suggested price to take profit for a BUY order.")
    stop_loss_price: Optional[float] = Field(None, description="Suggested price to set a stop-loss for a BUY order.")

class NewsSentiment(BaseModel):
    """Model for storing news sentiment analysis."""
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(..., description="Overall sentiment from the news.")
    summary: str = Field(..., description="A 1-2 sentence summary of the key news or narrative.")
    source_urls: Optional[List[str]] = Field(None, description="List of top 1-2 source URLs.")

class FinalInvestmentPlan(BaseModel):
    # ... (no changes needed) ...
    strategy_summary: str = Field(..., description="Brief summary of the overall investment strategy.")
    confidence_score: int = Field(..., ge=1, le=10, description="Confidence score (1-10) for this plan.")
    trade_recommendations: List[ActionableTrade] = Field(..., description="List of specific, actionable trades.")
    projected_portfolio_impact: str = Field(..., description="Expected impact on the portfolio's risk and return.")
    investment_timeline: str = Field(..., description="Suggested timeline for this plan (e.g., 'Short-term (1-4 weeks)').")

class ChatResponse(BaseModel):
    """Model for AI chat responses."""
    response: str = Field(..., description="The AI assistant's response to the user's question.")
    suggested_actions: Optional[List[str]] = Field(None, description="List of suggested follow-up actions.")
    
class MarketInsight(BaseModel):
    """Model for market analysis insights."""
    trend: Literal["Bullish", "Bearish", "Neutral"] = Field(..., description="Overall market trend.")
    key_levels: List[str] = Field(..., description="Important support/resistance levels.")
    market_summary: str = Field(..., description="Brief market summary and outlook.")

class PortfolioHealthScore(BaseModel):
    """Model for portfolio health assessment."""
    overall_score: int = Field(..., ge=0, le=100, description="Overall portfolio health score (0-100).")
    diversification_score: int = Field(..., ge=0, le=100, description="Portfolio diversification score.")
    performance_score: int = Field(..., ge=0, le=100, description="Recent performance score.")
    risk_score: int = Field(..., ge=0, le=100, description="Risk management score.")
    sentiment_score: int = Field(..., ge=0, le=100, description="News sentiment score.")
    recommendations: List[str] = Field(..., description="Specific recommendations to improve portfolio health.")

class CryptoPriceAnalysis(BaseModel):
    """Model for individual crypto price analysis and recommendations."""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    current_price: float = Field(..., description="Current market price in USD")
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"] = Field(..., description="Overall recommendation")
    confidence_score: int = Field(..., ge=1, le=10, description="Confidence in the recommendation (1-10)")
    
    # Price levels
    support_level: Optional[float] = Field(None, description="Key support level to watch")
    resistance_level: Optional[float] = Field(None, description="Key resistance level to watch")
    buy_price: Optional[float] = Field(None, description="Recommended buy price")
    sell_price: Optional[float] = Field(None, description="Recommended sell price")
    stop_loss: Optional[float] = Field(None, description="Recommended stop loss level")
    take_profit: Optional[float] = Field(None, description="Recommended take profit level")
    
    # Position sizing
    max_position_size_usd: Optional[float] = Field(None, description="Maximum recommended position size in USD")
    risk_percentage: Optional[float] = Field(None, description="Recommended risk percentage of portfolio")
    
    # Analysis reasoning
    technical_analysis: str = Field(..., description="Technical analysis summary")
    fundamental_analysis: str = Field(..., description="Fundamental analysis summary")
    risk_factors: List[str] = Field(..., description="Key risk factors to consider")
    time_horizon: str = Field(..., description="Recommended investment time horizon")

class InvestmentSearchQueries(BaseModel):
    """Model for generating targeted investment research queries."""
    trending_opportunities: List[str] = Field(..., description="Queries for trending investment opportunities")
    sector_analysis: List[str] = Field(..., description="Queries for specific crypto sectors (DeFi, GameFi, AI, etc.)")
    market_events: List[str] = Field(..., description="Queries for recent market events and news")
    technical_signals: List[str] = Field(..., description="Queries for technical analysis signals")
    fundamental_catalysts: List[str] = Field(..., description="Queries for fundamental catalysts and developments")

class InvestmentRecommendation(BaseModel):
    """Model for AI-generated investment recommendations based on internet research."""
    crypto_symbol: str = Field(..., description="Cryptocurrency symbol/name")
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "AVOID"] = Field(..., description="Investment recommendation")
    confidence_score: int = Field(..., ge=1, le=10, description="Confidence in recommendation (1-10)")
    price_target: Optional[float] = Field(None, description="Price target if available")
    investment_thesis: str = Field(..., description="Core investment thesis")
    catalysts: List[str] = Field(..., description="Key catalysts driving the recommendation")
    risks: List[str] = Field(..., description="Main risks to consider")
    time_horizon: str = Field(..., description="Recommended investment timeframe")
    
class MarketResearchReport(BaseModel):
    """Model for comprehensive market research report."""
    report_date: str = Field(..., description="Date of the research report")
    market_overview: str = Field(..., description="Overall market sentiment and trends")
    top_opportunities: List[InvestmentRecommendation] = Field(..., description="Top investment opportunities")
    sectors_to_watch: List[str] = Field(..., description="Crypto sectors showing promise")
    market_risks: List[str] = Field(..., description="Key market risks to monitor")
    research_summary: str = Field(..., description="Executive summary of research findings")

# --- API Clients and Helper Functions ---
cg = CoinGeckoAPI()
# Tavily client will be initialized dynamically based on available API key

@st.cache_data
def get_coin_id_map():
    try:
        coins = cg.get_coins_list()
        return {c['symbol'].upper(): c['id'] for c in coins}
    except Exception as e:
        st.warning(f"Could not fetch CoinGecko coin list: {e}")
        return {}

COIN_ID_MAP = get_coin_id_map()

def get_tavily_client(api_key):
    """Get Tavily client with the provided API key."""
    if api_key:
        try:
            return TavilyClient(api_key=api_key)
        except Exception as e:
            st.warning(f"Could not initialize Tavily client: {e}")
            return None
    return None

def get_llm_response(prompt, api_key, response_model):
    """Get structured response from OpenAI using instructor."""
    try:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        # Debug: Show what we're trying to process
        print(f"Debug - Processing prompt for model: {response_model.__name__}")
        
        openai_client = instructor.patch(OpenAI(api_key=api_key))
        response = openai_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[
                {"role": "system", "content": "You are a world-class financial analyst and cryptocurrency expert. Provide structured, data-driven analysis based on the user's request, conforming strictly to the provided Pydantic model schema. Think step-by-step to provide high-quality, reasoned outputs."},
                {"role": "user", "content": prompt}
            ],
            response_model=response_model,
            temperature=0.7,
            max_tokens=1500
        )
        print(f"Debug - Successfully got response for {response_model.__name__}")
        return response
    except Exception as e:
        error_msg = f"LLM API error: {str(e)}"
        print(f"Debug - {error_msg}")  # For debugging
        print(f"Debug - Error type: {type(e).__name__}")
        print(f"Debug - Full error: {repr(e)}")
        
        # Don't show error in UI for better user experience, log it instead
        # st.error(error_msg)
        return None

### NEW: Intelligent Investment Research Functions ###
def generate_investment_search_queries(risk_profile="Balanced"):
    """Generate intelligent search queries for investment research."""
    current_date = datetime.now()
    current_month = current_date.strftime("%B")
    current_year = current_date.year
    
    query_prompt = f"""
    Generate comprehensive search queries for cryptocurrency investment research. 
    Current date: {current_month} {current_year}
    Risk profile: {risk_profile}
    
    Create specific, targeted search queries that will help identify the best crypto investment opportunities. 
    Include current date context in queries to get recent information.
    
    Focus on:
    - Trending cryptocurrencies and emerging opportunities
    - Sector-specific analysis (DeFi, GameFi, AI tokens, Layer 2, etc.)
    - Recent market events and news that could impact investments
    - Technical analysis signals and patterns
    - Fundamental catalysts and upcoming developments
    
    Make queries specific and actionable for investment research in {current_month} {current_year}.
    """
    
    try:
        if not st.session_state.get('openai_api_key'):
            return None
        return get_llm_response(query_prompt, st.session_state.openai_api_key, InvestmentSearchQueries)
    except Exception as e:
        st.error(f"Could not generate search queries: {e}")
        return None

def conduct_investment_research(search_queries, max_results_per_query=3):
    """Conduct comprehensive investment research using Tavily search."""
    tavily_client = st.session_state.get('tavily_client')
    if not tavily_client or not search_queries:
        return None
    
    research_data = []
    
    # Combine all query categories
    all_queries = (
        search_queries.trending_opportunities +
        search_queries.sector_analysis +
        search_queries.market_events +
        search_queries.technical_signals +
        search_queries.fundamental_catalysts
    )
    
    with st.status("🔍 Conducting investment research...", expanded=True) as status:
        for i, query in enumerate(all_queries):
            try:
                st.write(f"Searching: {query}")
                
                # Perform search with advanced parameters
                search_results = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results_per_query,
                    include_domains=["coindesk.com", "cointelegraph.com", "decrypt.co", "theblock.co", "cryptoslate.com"]
                )
                
                for result in search_results.get('results', []):
                    research_data.append({
                        'query': query,
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': result.get('url', ''),
                        'published_date': result.get('published_date', '')
                    })
                
                # Small delay to respect API limits
                time.sleep(0.5)
                
            except Exception as e:
                st.write(f"Search failed for: {query[:50]}... - {e}")
                continue
        
        status.update(label="✅ Investment research completed!", state="complete")
    
    return research_data

def analyze_research_data(research_data, risk_profile="Balanced"):
    """Analyze research data and generate investment recommendations."""
    if not research_data or not st.session_state.get('openai_api_key'):
        return None
    
    # Compile research context
    research_context = ""
    for item in research_data:
        research_context += f"""
        Query: {item['query']}
        Title: {item['title']}
        Content: {item['content'][:500]}...
        URL: {item['url']}
        ---
        """
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    analysis_prompt = f"""
    You are a professional cryptocurrency investment analyst. Based on the comprehensive market research data below, 
    generate actionable investment recommendations.

    Research Date: {current_date}
    Risk Profile: {risk_profile}
    
    RESEARCH DATA:
    {research_context}
    
    Based on this research, provide:
    1. Overall market overview and sentiment
    2. Top 3-5 cryptocurrency investment opportunities with specific recommendations
    3. Key sectors showing promise
    4. Main market risks to consider
    5. Executive summary of findings
    
    Focus on actionable insights and specific cryptocurrencies that investors should consider.
    Be realistic about risks and provide balanced analysis.
    """
    
    try:
        return get_llm_response(analysis_prompt, st.session_state.openai_api_key, MarketResearchReport)
    except Exception as e:
        st.error(f"Could not analyze research data: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_smart_investment_recommendations(risk_profile="Balanced"):
    """Get AI-powered investment recommendations based on internet research."""
    # Generate search queries
    search_queries = generate_investment_search_queries(risk_profile)
    if not search_queries:
        return None, None, None
    
    # Conduct research
    research_data = conduct_investment_research(search_queries)
    if not research_data:
        return None, None, None
    
    # Analyze and generate recommendations
    analysis_report = analyze_research_data(research_data, risk_profile)
    
    # Return both the analysis report and the raw research data for transparency
    return analysis_report, research_data, search_queries

### NEWS Analysis Agent function ###
def get_news_analysis(coin_symbol: str, coin_name: str) -> Optional[NewsSentiment]:
    """Uses Tavily to search for recent news and an LLM to analyze it."""
    # Check if we already have cached news for this coin in the session
    if hasattr(st.session_state, 'news_cache') and coin_symbol in st.session_state.news_cache:
        return st.session_state.news_cache[coin_symbol]
    
    tavily_client = st.session_state.get('tavily_client')
    if not tavily_client or not st.session_state.get('openai_api_key'):
        return None
    try:
        search_query = f"latest cryptocurrency news and developments for {coin_name} ({coin_symbol})"
        search_results = tavily_client.search(query=search_query, search_depth="advanced")
        context = " ".join([res['content'] for res in search_results['results'][:4]])

        if not context.strip():
            return None

        prompt = f"""
        Based on the following news articles, analyze the current sentiment for {coin_name} ({coin_symbol}).
        Summarize the key points and determine if the overall sentiment is Positive, Negative, or Neutral.
        News context:
        ---
        {context}
        ---
        """
        news_analysis = get_llm_response(prompt, st.session_state.openai_api_key, NewsSentiment)
        if news_analysis:
            # Add sources from the original Tavily search
            news_analysis.source_urls = [res['url'] for res in search_results['results'][:2]]
        
        # Cache the result in session state
        if not hasattr(st.session_state, 'news_cache'):
            st.session_state.news_cache = {}
        st.session_state.news_cache[coin_symbol] = news_analysis
        
        return news_analysis

    except Exception as e:
        st.warning(f"Could not perform news analysis for {coin_symbol}: {e}")
        return None
### NEW: Enhanced Market Data Functions ###
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_overview():
    """Get overall crypto market data and trending coins."""
    try:
        # Global market data
        global_data = cg.get_global()
        
        # Check if global_data is valid
        if not global_data:
            st.warning("No global market data available")
            return None
        
        # Top trending coins
        trending = cg.get_search_trending()
        
        # Top gainers/losers
        coins_market = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=50,  # Reduced from CONFIG value to avoid API limits
            page=1,
            sparkline=True,
            price_change_percentage='24h,7d'
        )
        
        df_market = pd.DataFrame(coins_market)
        
        # Extract data directly (CoinGecko API doesn't use nested 'data' structure)
        total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
        market_cap_change_24h = global_data.get('market_cap_change_percentage_24h_usd', 0)
        bitcoin_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
        
        return {
            "global_data": global_data,
            "trending": trending,
            "market_data": df_market,
            "total_market_cap": total_market_cap,
            "market_cap_change_24h": market_cap_change_24h,
            "bitcoin_dominance": bitcoin_dominance
        }
    except Exception as e:
        st.warning(f"Could not fetch market overview: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_crypto_screener(sort_by="market_cap_desc", category="", min_market_cap=None):
    """Advanced crypto screener with filters."""
    try:
        params = {
            'vs_currency': 'usd',
            'order': sort_by,
            'per_page': 50,
            'page': 1,
            'sparkline': True,
            'price_change_percentage': '1h,24h,7d,30d'
        }
        
        if category:
            params['category'] = category
            
        coins = cg.get_coins_markets(**params)
        df = pd.DataFrame(coins)
        
        if min_market_cap and not df.empty:
            df = df[df['market_cap'] >= min_market_cap]
            
        return df
        
    except Exception as e:
        st.warning(f"Could not fetch screener data: {e}")
        return pd.DataFrame()

def calculate_portfolio_health(portfolio_df):
    """Calculate comprehensive portfolio health score."""
    if portfolio_df.empty:
        return None
        
    try:
        # Diversification Score (based on number of assets and distribution)
        num_assets = len(portfolio_df)
        total_value = portfolio_df['USD Value'].sum()
        largest_holding_pct = portfolio_df['USD Value'].max() / total_value * 100
        
        diversification_score = min(100, (num_assets * 10) + (100 - largest_holding_pct))
        
        # Performance Score (based on 24h and 7d performance)
        avg_24h_perf = portfolio_df['24h Perf (%)'].mean()
        avg_7d_perf = portfolio_df['7d Perf (%)'].mean()
        performance_score = max(0, min(100, 50 + (avg_24h_perf + avg_7d_perf) * 2))
        
        # Risk Score (based on RSI and volatility)
        avg_rsi = portfolio_df['RSI'].dropna().mean()
        risk_score = 100 - abs(avg_rsi - 50) * 2  # Closer to 50 RSI is better
        
        # Sentiment Score (based on news sentiment)
        positive_sentiment = len(portfolio_df[portfolio_df['News Sentiment'] == 'Positive'])
        total_with_news = len(portfolio_df[portfolio_df['News Sentiment'] != 'N/A'])
        sentiment_score = (positive_sentiment / max(1, total_with_news)) * 100
        
        # Overall Score (weighted average)
        weights = CONFIG["portfolio_health_weights"]
        overall_score = (
            diversification_score * weights["diversification"] +
            performance_score * weights["performance"] +
            risk_score * weights["risk_management"] +
            sentiment_score * weights["sentiment"]
        )
        
        return {
            "overall_score": int(overall_score),
            "diversification_score": int(diversification_score),
            "performance_score": int(performance_score),
            "risk_score": int(risk_score),
            "sentiment_score": int(sentiment_score),
            "recommendations": generate_health_recommendations(
                diversification_score, performance_score, risk_score, sentiment_score
            )
        }
        
    except Exception as e:
        st.warning(f"Could not calculate portfolio health: {e}")
        return None

def generate_health_recommendations(div_score, perf_score, risk_score, sent_score):
    """Generate actionable recommendations based on health scores."""
    recommendations = []
    
    if div_score < 60:
        recommendations.append("Consider diversifying into more assets to reduce concentration risk")
    if perf_score < 40:
        recommendations.append("Review underperforming assets and consider rebalancing")
    if risk_score < 50:
        recommendations.append("Monitor RSI levels - some assets may be overbought/oversold")
    if sent_score < 40:
        recommendations.append("Pay attention to negative news sentiment affecting your holdings")
        
    if not recommendations:
        recommendations.append("Portfolio health looks good! Continue monitoring market conditions")
        
    return recommendations

def get_fallback_response(user_question, portfolio_context=None):
    """Provide basic responses for common questions when AI is unavailable."""
    question_lower = user_question.lower()
    
    # Basic portfolio responses
    if portfolio_context is not None and not portfolio_context.empty:
        if any(word in question_lower for word in ['portfolio', 'holdings', 'performance']):
            total_value = portfolio_context['USD Value'].sum()
            num_assets = len(portfolio_context)
            return f"Based on your portfolio data: You have {num_assets} assets with a total value of ${total_value:,.2f}. For detailed analysis, please ensure your OpenAI API key is working."
        
        if any(word in question_lower for word in ['sell', 'selling']):
            return "For selling recommendations, I need the AI analysis to be working. Please check your API key and try again."
        
        if any(word in question_lower for word in ['buy', 'buying', 'invest']):
            return "For investment recommendations, I need the AI analysis to be working. Please check your API key and try again."
    
    # General crypto responses
    if any(word in question_lower for word in ['bitcoin', 'btc']):
        return "Bitcoin is the first and largest cryptocurrency. For current analysis and recommendations, please ensure your OpenAI API key is configured correctly."
    
    if any(word in question_lower for word in ['ethereum', 'eth']):
        return "Ethereum is a smart contract platform and the second-largest cryptocurrency. For detailed analysis, please check your API key configuration."
    
    # Default response
    return None

def get_ai_chat_response(user_question, portfolio_context=None):
    """AI chat assistant for crypto advice with comprehensive portfolio context."""
    try:
        # Validate API key first
        api_key = st.session_state.get('openai_api_key')
        if not api_key:
            return None
        
        # Build comprehensive context information
        context = ""
        portfolio_analysis = ""
        
        if portfolio_context is not None and not portfolio_context.empty:
            # Detailed portfolio analysis
            total_value = portfolio_context['USD Value'].sum()
            num_assets = len(portfolio_context)
            
            # Top holdings analysis
            top_holdings = portfolio_context.nlargest(5, 'USD Value')[['Asset', 'Name', 'USD Value', '24h Perf (%)', '7d Perf (%)', 'RSI', 'News Sentiment']]
            
            # Performance analysis
            total_24h_change = (portfolio_context['USD Value'] * portfolio_context['24h Perf (%)'] / 100).sum()
            portfolio_24h_perf = (total_24h_change / total_value) * 100 if total_value > 0 else 0
            
            total_7d_change = (portfolio_context['USD Value'] * portfolio_context['7d Perf (%)'] / 100).sum()
            portfolio_7d_perf = (total_7d_change / total_value) * 100 if total_value > 0 else 0
            
            # Risk analysis
            avg_rsi = portfolio_context['RSI'].dropna().mean()
            overbought_assets = len(portfolio_context[portfolio_context['RSI'] > 70])
            oversold_assets = len(portfolio_context[portfolio_context['RSI'] < 30])
            
            # Sentiment analysis
            positive_sentiment = len(portfolio_context[portfolio_context['News Sentiment'] == 'Positive'])
            negative_sentiment = len(portfolio_context[portfolio_context['News Sentiment'] == 'Negative'])
            neutral_sentiment = len(portfolio_context[portfolio_context['News Sentiment'] == 'Neutral'])
            
            # Diversification analysis
            largest_holding_pct = (portfolio_context['USD Value'].max() / total_value) * 100
            concentration_risk = "High" if largest_holding_pct > 50 else "Medium" if largest_holding_pct > 30 else "Low"
            
            # Sector analysis (basic categorization)
            major_coins = ['BTC', 'ETH']
            defi_coins = ['UNI', 'AAVE', 'COMP', 'SUSHI', 'CAKE']
            layer1_coins = ['ADA', 'SOL', 'DOT', 'AVAX', 'NEAR', 'ALGO']
            
            btc_eth_allocation = portfolio_context[portfolio_context['Asset'].isin(major_coins)]['USD Value'].sum()
            defi_allocation = portfolio_context[portfolio_context['Asset'].isin(defi_coins)]['USD Value'].sum()
            layer1_allocation = portfolio_context[portfolio_context['Asset'].isin(layer1_coins)]['USD Value'].sum()
            
            btc_eth_pct = (btc_eth_allocation / total_value) * 100 if total_value > 0 else 0
            defi_pct = (defi_allocation / total_value) * 100 if total_value > 0 else 0
            layer1_pct = (layer1_allocation / total_value) * 100 if total_value > 0 else 0
            
            portfolio_analysis = f"""

COMPREHENSIVE PORTFOLIO ANALYSIS:

📊 PORTFOLIO OVERVIEW:
- Total Portfolio Value: ${total_value:,.2f}
- Number of Assets: {num_assets}
- 24h Performance: {portfolio_24h_perf:+.2f}%
- 7d Performance: {portfolio_7d_perf:+.2f}%
- Risk Profile: {st.session_state.get('risk_profile', 'Balanced')}

🏆 TOP 5 HOLDINGS:
"""
            for i, (_, row) in enumerate(top_holdings.iterrows(), 1):
                allocation_pct = (row['USD Value'] / total_value) * 100
                portfolio_analysis += f"""
{i}. {row['Asset']} ({row['Name']}):
   • Value: ${row['USD Value']:,.2f} ({allocation_pct:.1f}% of portfolio)
   • 24h: {row['24h Perf (%)']:+.2f}% | 7d: {row['7d Perf (%)']:+.2f}%
   • RSI: {row['RSI']:.1f if pd.notna(row['RSI']) else 'N/A'}
   • News Sentiment: {row['News Sentiment']}
"""
            
            portfolio_analysis += f"""

⚠️ RISK ANALYSIS:
- Average RSI: {avg_rsi:.1f if pd.notna(avg_rsi) else 'N/A'}
- Overbought Assets (RSI >70): {overbought_assets}
- Oversold Assets (RSI <30): {oversold_assets}
- Concentration Risk: {concentration_risk} (Largest holding: {largest_holding_pct:.1f}%)

📰 SENTIMENT OVERVIEW:
- Positive News: {positive_sentiment} assets
- Negative News: {negative_sentiment} assets  
- Neutral News: {neutral_sentiment} assets

🏗️ SECTOR ALLOCATION:
- BTC/ETH (Major): {btc_eth_pct:.1f}%
- DeFi Tokens: {defi_pct:.1f}%
- Layer 1 Platforms: {layer1_pct:.1f}%
- Other/Alt Coins: {100 - btc_eth_pct - defi_pct - layer1_pct:.1f}%

📈 PERFORMANCE INSIGHTS:
- Best Performer (24h): {portfolio_context.loc[portfolio_context['24h Perf (%)'].idxmax(), 'Asset']} ({portfolio_context['24h Perf (%)'].max():+.2f}%)
- Worst Performer (24h): {portfolio_context.loc[portfolio_context['24h Perf (%)'].idxmin(), 'Asset']} ({portfolio_context['24h Perf (%)'].min():+.2f}%)
"""
            
            context = portfolio_analysis
        else:
            context = "\n⚠️ No portfolio data available. User hasn't connected their Binance account or portfolio analysis hasn't been run yet.\n"
        
        # Get current date and market context
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Enhanced prompt with portfolio-aware instructions
        prompt = f"""
You are an expert cryptocurrency investment advisor and portfolio manager. You have access to the user's complete portfolio information and should provide PERSONALIZED, SPECIFIC advice based on their actual holdings and performance.

Current Date: {current_date}
{context}

IMPORTANT INSTRUCTIONS:
1. If the user has a portfolio, ALWAYS reference their specific holdings, performance, and allocation when answering
2. Provide specific recommendations based on their actual assets and risk levels
3. Point out opportunities and risks in their current holdings
4. Suggest rebalancing, profit-taking, or position adjustments based on their portfolio
5. If they ask generic questions, still relate answers back to their portfolio when relevant
6. Use specific numbers from their portfolio (values, percentages, RSI levels, etc.)
7. Consider their risk profile and current allocation when making suggestions

User Question: {user_question}

Provide a comprehensive, personalized response that:
- Directly answers their question with specific reference to their portfolio when applicable
- Highlights relevant insights from their current holdings
- Provides actionable recommendations based on their actual positions
- Considers their portfolio performance and risk metrics
- Suggests 2-3 specific follow-up actions they could take

Be conversational but professional. Focus on PERSONALIZED advice, not generic responses.
"""
        
        response = get_llm_response(prompt, api_key, ChatResponse)
        return response
        
    except Exception as e:
        error_msg = f"Chat assistant error: {e}"
        st.error(error_msg)
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_crypto_technical_analysis(symbol, vs_currency='usd'):
    """Get detailed technical analysis data for a specific cryptocurrency."""
    try:
        # Get current market data
        coin_data = cg.get_coins_markets(vs_currency=vs_currency, ids=symbol, sparkline=True, price_change_percentage='1h,24h,7d,30d')[0]
        
        # Get historical price data (30 days)
        historical_data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency=vs_currency, days=30)
        
        # Convert to DataFrame for analysis
        prices = pd.DataFrame(historical_data['prices'], columns=['timestamp', 'price'])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators
        prices['sma_7'] = prices['price'].rolling(window=7).mean()
        prices['sma_14'] = prices['price'].rolling(window=14).mean()
        prices['sma_21'] = prices['price'].rolling(window=21).mean()
        
        # Calculate RSI
        delta = prices['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate support and resistance levels
        recent_prices = prices['price'].tail(14)
        support_level = recent_prices.min() * 0.98  # 2% below recent low
        resistance_level = recent_prices.max() * 1.02  # 2% above recent high
        
        current_price = coin_data['current_price']
        
        return {
            'coin_data': coin_data,
            'prices_df': prices,
            'current_price': current_price,
            'rsi': rsi.iloc[-1] if not rsi.empty else None,
            'sma_7': prices['sma_7'].iloc[-1] if not prices['sma_7'].empty else None,
            'sma_14': prices['sma_14'].iloc[-1] if not prices['sma_14'].empty else None,
            'sma_21': prices['sma_21'].iloc[-1] if not prices['sma_21'].empty else None,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'volume_24h': coin_data['total_volume'],
            'market_cap': coin_data['market_cap'],
            'price_change_24h': coin_data['price_change_percentage_24h'],
            'price_change_7d': coin_data['price_change_percentage_7d_in_currency'],
        }
        
    except Exception as e:
        st.warning(f"Could not fetch technical analysis for {symbol}: {e}")
        return None

def get_crypto_price_analysis(symbol, risk_profile="Balanced", portfolio_size_usd=10000):
    """Get comprehensive price analysis and recommendations for a specific crypto."""
    try:
        # Get technical analysis data
        tech_data = get_crypto_technical_analysis(symbol)
        if not tech_data:
            return None
        
        # Get news sentiment
        coin_name = tech_data['coin_data']['name']
        news_analysis = get_news_analysis(symbol.upper(), coin_name)
        
        # Prepare analysis context
        analysis_context = f"""
        CRYPTOCURRENCY ANALYSIS REQUEST
        
        Symbol: {symbol.upper()}
        Name: {coin_name}
        Current Price: ${tech_data['current_price']:,.4f}
        Market Cap: ${tech_data['market_cap']:,.0f}
        24h Volume: ${tech_data['volume_24h']:,.0f}
        
        TECHNICAL INDICATORS:
        - RSI: {tech_data['rsi']:.2f if tech_data['rsi'] else 'N/A'}
        - Price vs SMA 7: {(tech_data['current_price'] / tech_data['sma_7'] - 1) * 100:.2f}% if tech_data['sma_7'] else 'N/A'
        - Price vs SMA 14: {(tech_data['current_price'] / tech_data['sma_14'] - 1) * 100:.2f}% if tech_data['sma_14'] else 'N/A'
        - Price vs SMA 21: {(tech_data['current_price'] / tech_data['sma_21'] - 1) * 100:.2f}% if tech_data['sma_21'] else 'N/A'
        - Support Level: ${tech_data['support_level']:.4f}
        - Resistance Level: ${tech_data['resistance_level']:.4f}
        
        PERFORMANCE:
        - 24h Change: {tech_data['price_change_24h']:.2f}%
        - 7d Change: {tech_data['price_change_7d']:.2f}% if tech_data['price_change_7d'] else 'N/A'
        
        NEWS SENTIMENT: {news_analysis.sentiment if news_analysis else 'N/A'}
        NEWS SUMMARY: {news_analysis.summary if news_analysis else 'No recent news available'}
        
        USER PROFILE:
        - Risk Profile: {risk_profile}
        - Portfolio Size: ${portfolio_size_usd:,.0f}
        
        Please provide a comprehensive analysis with specific buy/sell recommendations, price levels, and position sizing.
        """
        
        analysis_prompt = f"""
        You are a professional cryptocurrency analyst. Based on the provided data, give a comprehensive analysis and trading recommendation.
        
        {analysis_context}
        
        Provide specific, actionable recommendations including:
        1. Clear BUY/SELL/HOLD recommendation with confidence level
        2. Specific entry/exit price levels
        3. Position sizing based on risk profile
        4. Stop loss and take profit levels
        5. Technical and fundamental reasoning
        6. Key risk factors
        7. Investment time horizon
        
        Be precise with numbers and realistic with expectations.
        """
        
        return get_llm_response(analysis_prompt, st.session_state.openai_api_key, CryptoPriceAnalysis)
        
    except Exception as e:
        st.error(f"Could not generate price analysis: {e}")
        return None

### REFACTORED: `get_advanced_binance_data` broken into smaller functions ###
def find_and_get_price(client, coin):
    """Finds a valid trading pair and gets the latest price."""
    if coin in ["USDT", "BUSD", "USDC", "FDUSD"]:
        return 1.0, None # It's a stablecoin, price is 1
    
    for quote in CONFIG['quote_currencies']:
        symbol = f"{coin}{quote}"
        try:
            price_info = client.get_symbol_ticker(symbol=symbol)
            price = float(price_info['price'])
            # If quote is not USD-based, convert it
            if quote not in ["USDT", "BUSD", "USDC", "FDUSD"]:
                quote_price_in_usd = float(client.get_symbol_ticker(symbol=f"{quote}USDT")['price'])
                price *= quote_price_in_usd
            return price, symbol
        except BinanceAPIException as e:
            if e.code == -1121: # Invalid symbol
                continue
            else:
                raise e # Re-raise other API errors
    return None, None # No valid pair found

def get_asset_technical_data(client, symbol):
    """Fetches historical klines and calculates TA indicators."""
    klines = client.get_historical_klines(symbol, CONFIG["kline_interval_ta"], CONFIG["kline_history_ta"])
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    for col in ['close', 'high', 'low', 'open', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    rsi = df.ta.rsi().iloc[-1] if not df.ta.rsi().empty else None
    sma50 = df.ta.sma(50).iloc[-1] if not df.ta.sma(50).empty else None
    sma200 = df.ta.sma(200).iloc[-1] if not df.ta.sma(200).empty else None
    return {"rsi": rsi, "sma50": sma50, "sma200": sma200, "dataframe": df}

def get_asset_performance_data(client, symbol):
    """Fetches daily klines and calculates performance and sparkline."""
    klines = client.get_historical_klines(symbol, CONFIG["kline_interval_perf"], CONFIG["kline_history_perf"])
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['close'] = pd.to_numeric(df['close'])
    
    perf_24h = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0
    perf_7d = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100 if len(df) > 0 else 0
    sparkline = create_sparkline(df['close'])
    return {"perf_24h": perf_24h, "perf_7d": perf_7d, "sparkline": sparkline}

@st.cache_data(ttl=300)
def get_full_portfolio_analysis(_binance_client, _openai_key, _tavily_key):
    """Fetches comprehensive portfolio, technical, and fundamental data."""
    try:
        client = Client(_binance_client, _openai_key)
        account_info = client.get_account()
        balances = [b for b in account_info['balances'] if float(b['free']) > 0.00001]
        
        portfolio_data = []
        total_portfolio_value = 0
        usdt_balance = 0

        # Initialize news cache if it doesn't exist
        if not hasattr(st.session_state, 'news_cache'):
            st.session_state.news_cache = {}

        for i, asset in enumerate(balances):
            coin = asset['asset']
            balance = float(asset['free'])

            # Skip leveraged tokens and certain stablecoins early
            if coin.endswith(('UP', 'DOWN', 'BEAR', 'BULL')): continue
            if coin in ['USDT', 'USDC', 'FDUSD', 'BUSD']:
                usd_value = balance
                usdt_balance += usd_value # Aggregate all stablecoins
                total_portfolio_value += usd_value
                continue # Skip detailed analysis for stables

            price, symbol = find_and_get_price(client, coin)
            if not price or not symbol: continue
            
            usd_value = balance * price
            if usd_value < CONFIG["min_usd_value_display"]: continue
            total_portfolio_value += usd_value

            with st.status(f"Analyzing {coin} ({i+1}/{len(balances)})...", expanded=False) as status:
                st.write(f"Finding price for {coin}...")
                tech_data = get_asset_technical_data(client, symbol)
                st.write(f"Calculating technicals for {symbol}...")
                perf_data = get_asset_performance_data(client, symbol)
                st.write(f"Calculating performance for {symbol}...")
                
                coin_id = COIN_ID_MAP.get(coin)
                market_cap = None
                full_name = coin
                if coin_id:
                    coin_info = cg.get_coin_by_id(coin_id, market_data='true', community_data=False, developer_data=False, sparkline=False)
                    market_cap = coin_info.get('market_data', {}).get('market_cap', {}).get('usd')
                    full_name = coin_info.get('name', coin)
                    st.write(f"Fetching fundamental data for {full_name}...")

                ### NEW: Integrate News Analysis ###
                news_analysis = get_news_analysis(coin, full_name)
                st.write(f"Analyzing news sentiment for {full_name}...")

                portfolio_data.append({
                    "Asset": coin, "Name": full_name, "Balance": balance, "Price": price, "USD Value": usd_value,
                    "24h Perf (%)": perf_data["perf_24h"], "7d Perf (%)": perf_data["perf_7d"],
                    "RSI": tech_data["rsi"],
                    "Price/SMA50": price / tech_data["sma50"] if tech_data["sma50"] and tech_data["sma50"] > 0 else None,
                    "SMA50/SMA200": tech_data["sma50"] / tech_data["sma200"] if tech_data["sma50"] and tech_data["sma200"] and tech_data["sma200"] > 0 else None,
                    "Market Cap": market_cap, "Sparkline": perf_data["sparkline"],
                    "News Sentiment": news_analysis.sentiment if news_analysis else "N/A",
                    "News Summary": news_analysis.summary if news_analysis else "No recent news found."
                })
                status.update(label=f"Analysis for {coin} complete!", state="complete")
                time.sleep(0.1) # ### NEW: Small delay to respect API limits

        if not portfolio_data: return None, None
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_summary = {
            "total_value_usd": total_portfolio_value, 
            "dataframe": portfolio_df,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") ### NEW: Data freshness timestamp
        }
        
        return portfolio_summary, usdt_balance

    except BinanceAPIException as e:
        st.error(f"A Binance API error occurred: {e}. Please check your keys and permissions.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching: {e}")
        return None, None

# ... (create_sparkline, create_binance_trade_link, generate_pdf remain the same) ...
def create_sparkline(data):
    """Creates a base64-encoded sparkline image."""
    fig, ax = plt.subplots(1, 1, figsize=(2, 0.5))
    ax.plot(data, color='green' if data.iloc[-1] >= data.iloc[0] else 'red', linewidth=0.8)
    ax.set_axis_off()
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def create_binance_trade_link(symbol, action, usd_amount, price):
    """Constructs a URL to pre-fill a trade on Binance."""
    if action not in ["BUY", "SELL"] or not usd_amount or not price or price == 0:
        return None
    
    quantity = usd_amount / price
    base_asset = symbol.replace("USDT", "")
    
    trade_url = f"https://www.binance.com/en/trade/{base_asset}_USDT?type=market&side={action.lower()}&amount={quantity:.6f}"
    return trade_url

def generate_pdf(plan, portfolio_summary, risk_profile):
    """Generates a PDF report of the analysis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    pdf.cell(0, 10, 'AI Investment Committee Report', 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')} | Risk Profile: {risk_profile}", 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Strategy Summary', 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, plan.strategy_summary)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Actionable Trades', 0, 1)
    pdf.set_font("Arial", '', 10)
    for trade in plan.trade_recommendations:
        pdf.multi_cell(0, 5, f"- {trade.action} {trade.coin} (${trade.amount_usd or 'N/A'}): {trade.reasoning}")
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Portfolio Snapshot', 0, 1)
    pdf.set_font("Arial", '', 8)
    for i, row in portfolio_summary['dataframe'].iterrows():
        pdf.cell(0, 5, f"{row['Asset']}: ${row['USD Value']:,.2f}", 0, 1)
        
    return pdf.output(dest='S').encode('latin1')
# --- Streamlit UI and Application Logic ---
st.title("📈 AI Investment Committee Pro")
st.markdown("### 🚀 Advanced Multi-Agent System with Deep Market Analysis")

# Create main navigation
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    if st.button("🏠 Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
with col2:
    if st.button("📊 Portfolio", use_container_width=True):
        st.session_state.page = "portfolio"
with col3:
    if st.button("🔍 Screener", use_container_width=True):
        st.session_state.page = "screener"
with col4:
    if st.button("💰 Price Analysis", use_container_width=True):
        st.session_state.page = "price_analysis"
with col5:
    if st.button("🧠 Smart Research", use_container_width=True):
        st.session_state.page = "smart_research"
with col6:
    if st.button("💬 AI Assistant", use_container_width=True):
        st.session_state.page = "chat"
with col7:
    if st.button("📚 Learn", use_container_width=True):
        st.session_state.page = "learn"

# Initialize page if not set
if 'page' not in st.session_state:
    st.session_state.page = "dashboard"

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key inputs
    openai_api_key_input = st.text_input("OpenAI API Key", type="password", help="platform.openai.com")
    binance_api_key_input = st.text_input("Binance API Key (Read-Only)", type="password")
    binance_api_secret_input = st.text_input("Binance API Secret (Read-Only)", type="password")
    tavily_api_key_input = st.text_input("Tavily API Key", type="password", help="app.tavily.com (for news analysis)")

    # Smart fallback: Use input if provided, otherwise use environment variables
    openai_api_key = openai_api_key_input or os.getenv("OPENAI_API_KEY")
    binance_api_key = binance_api_key_input or os.getenv("BINANCE_API_KEY")
    binance_api_secret = binance_api_secret_input or os.getenv("BINANCE_API_SECRET")
    tavily_api_key = tavily_api_key_input or os.getenv("TAVILY_API_KEY")
    
    # Store in session state for use across the app
    st.session_state.openai_api_key = openai_api_key
    st.session_state.binance_api_key = binance_api_key
    st.session_state.binance_api_secret = binance_api_secret
    st.session_state.tavily_api_key = tavily_api_key
    
    # Initialize Tavily client dynamically
    st.session_state.tavily_client = get_tavily_client(tavily_api_key)
    
    # Show which keys are loaded
    with st.expander("🔑 API Key Status", expanded=False):
        st.write(f"OpenAI API: {'✅ Configured' if openai_api_key else '❌ Missing'}")
        st.write(f"Binance API: {'✅ Configured' if binance_api_key else '❌ Missing'}")
        st.write(f"Binance Secret: {'✅ Configured' if binance_api_secret else '❌ Missing'}")
        st.write(f"Tavily API: {'✅ Configured' if tavily_api_key else '❌ Missing'}")
        
        if not openai_api_key_input and openai_api_key:
            st.info("🔄 OpenAI API loaded from environment")
        if not binance_api_key_input and binance_api_key:
            st.info("🔄 Binance API loaded from environment")
        if not binance_api_secret_input and binance_api_secret:
            st.info("🔄 Binance Secret loaded from environment")
        if not tavily_api_key_input and tavily_api_key:
            st.info("🔄 Tavily API loaded from environment")

    st.session_state.risk_profile = st.selectbox(
        "🎯 Risk Profile", CONFIG["risk_profiles"], index=1
    )
    
    st.divider()
    
    # Quick Market Stats in Sidebar
    st.subheader("📈 Quick Market Stats")
    market_data = get_market_overview()
    if market_data:
        st.metric(
            "Total Market Cap", 
            f"${market_data['total_market_cap']:,.0f}",
            f"{market_data['market_cap_change_24h']:+.2f}%"
        )
        st.metric(
            "Bitcoin Dominance", 
            f"{market_data['bitcoin_dominance']:.1f}%"
        )
    
    st.divider()
    
    # Main analysis button
    analyze_button = st.button("🚀 Analyze Portfolio", use_container_width=True, type="primary")
    if st.session_state.get('stage', 'initial') != 'initial':
        if st.button("🔄 New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()): 
                if key not in ['page']:  # Keep page state
                    del st.session_state[key]
            st.rerun()

# PAGE ROUTING
if st.session_state.page == "dashboard":
    ### DASHBOARD PAGE ###
    st.header("🏠 Market Dashboard")
    
    # Market Overview Section
    market_data = get_market_overview()
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Market Cap", 
                f"${market_data['total_market_cap']:,.0f}",
                f"{market_data['market_cap_change_24h']:+.2f}%"
            )
        with col2:
            st.metric("BTC Dominance", f"{market_data['bitcoin_dominance']:.1f}%")
        
        with col3:
            total_coins = len(market_data['market_data'])
            st.metric("Tracked Coins", f"{total_coins:,}")
            
        with col4:
            gainers = len(market_data['market_data'][market_data['market_data']['price_change_percentage_24h'] > 0])
            st.metric("24h Gainers", f"{gainers}/{total_coins}")
        
        # Trending Coins
        st.subheader("🔥 Trending Coins")
        trending_cols = st.columns(5)
        for i, coin in enumerate(market_data['trending']['coins'][:5]):
            with trending_cols[i]:
                st.info(f"**{coin['item']['name']}** ({coin['item']['symbol'].upper()})")
        
        # Top Gainers/Losers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top Gainers (24h)")
            gainers = market_data['market_data'].nlargest(5, 'price_change_percentage_24h')[['name', 'symbol', 'current_price', 'price_change_percentage_24h']]
            for _, row in gainers.iterrows():
                st.success(f"**{row['name']}** ({row['symbol'].upper()}) - ${row['current_price']:.4f} (+{row['price_change_percentage_24h']:.2f}%)")
        
        with col2:
            st.subheader("📉 Top Losers (24h)")
            losers = market_data['market_data'].nsmallest(5, 'price_change_percentage_24h')[['name', 'symbol', 'current_price', 'price_change_percentage_24h']]
            for _, row in losers.iterrows():
                st.error(f"**{row['name']}** ({row['symbol'].upper()}) - ${row['current_price']:.4f} ({row['price_change_percentage_24h']:.2f}%)")

elif st.session_state.page == "price_analysis":
    ### PRICE ANALYSIS PAGE ###
    st.header("💰 Crypto Price Analysis & Trading Signals")
    st.markdown("Get detailed buy/sell recommendations with specific price levels and position sizing for any cryptocurrency.")
    
    if not st.session_state.get('openai_api_key'):
        st.warning("⚠️ Please enter your OpenAI API key to use the AI price analysis.")
    else:
        # Input section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Popular cryptocurrencies for quick selection
            popular_cryptos = [
                'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 
                'polkadot', 'dogecoin', 'avalanche-2', 'chainlink', 'polygon',
                'litecoin', 'shiba-inu', 'cosmos', 'algorand', 'fantom'
            ]
            
            selected_crypto = st.selectbox(
                "🪙 Select Cryptocurrency", 
                options=popular_cryptos,
                format_func=lambda x: x.replace('-', ' ').title()
            )
        
        with col2:
            risk_profile = st.selectbox(
                "🎯 Your Risk Profile", 
                CONFIG["risk_profiles"],
                index=1
            )
        
        with col3:
            portfolio_size = st.number_input(
                "💼 Portfolio Size (USD)", 
                min_value=100, 
                value=10000, 
                step=500,
                help="Your total portfolio size for position sizing calculations"
            )
        
        # Custom crypto input
        st.markdown("---")
        custom_crypto = st.text_input(
            "🔍 Or enter any crypto symbol/name:", 
            placeholder="e.g., bitcoin, ethereum, matic-network",
            help="Use CoinGecko ID format (lowercase with hyphens)"
        )
        
        if custom_crypto:
            selected_crypto = custom_crypto.lower().strip()
        
        # Analysis button
        if st.button("🔬 Analyze Price & Get Trading Signals", type="primary", use_container_width=True):
            with st.spinner(f"🧠 AI is analyzing {selected_crypto.replace('-', ' ').title()}..."):
                analysis = get_crypto_price_analysis(selected_crypto, risk_profile, portfolio_size)
                
                if analysis:
                    st.session_state.current_analysis = analysis
                    st.session_state.analyzed_crypto = selected_crypto
                    st.success(f"✅ Analysis completed for {selected_crypto.replace('-', ' ').title()}!")
                else:
                    st.error(f"❌ Could not analyze {selected_crypto}. Please check the symbol and try again.")
        
        # Display analysis results
        if 'current_analysis' in st.session_state and 'analyzed_crypto' in st.session_state:
            analysis = st.session_state.current_analysis
            crypto_name = st.session_state.analyzed_crypto.replace('-', ' ').title()
            
            st.markdown("---")
            st.subheader(f"📋 Analysis Results for {crypto_name}")
            
            # Main recommendation
            rec_color = {
                "STRONG_BUY": "green",
                "BUY": "green", 
                "HOLD": "orange",
                "SELL": "red",
                "STRONG_SELL": "red"
            }
            
            rec_icon = {
                "STRONG_BUY": "🟢",
                "BUY": "🟢", 
                "HOLD": "🟡",
                "SELL": "🔴",
                "STRONG_SELL": "🔴"
            }
            
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"### {rec_icon[analysis.recommendation]} {analysis.recommendation}")
                    st.markdown(f"**Current Price:** ${analysis.current_price:,.4f}")
                with col2:
                    st.metric("Confidence Score", f"{analysis.confidence_score}/10")
                    st.markdown(f"**Time Horizon:** {analysis.time_horizon}")
                with col3:
                    if analysis.max_position_size_usd:
                        st.metric("Max Position Size", f"${analysis.max_position_size_usd:,.0f}")
                    if analysis.risk_percentage:
                        st.metric("Risk %", f"{analysis.risk_percentage:.1f}%")
            
            # Price levels
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Key Price Levels")
                if analysis.buy_price:
                    st.success(f"🎯 **Buy Price:** ${analysis.buy_price:,.4f}")
                if analysis.sell_price:
                    st.error(f"🎯 **Sell Price:** ${analysis.sell_price:,.4f}")
                if analysis.support_level:
                    st.info(f"🛡️ **Support:** ${analysis.support_level:,.4f}")
                if analysis.resistance_level:
                    st.warning(f"⚡ **Resistance:** ${analysis.resistance_level:,.4f}")
                if analysis.stop_loss:
                    st.error(f"🛑 **Stop Loss:** ${analysis.stop_loss:,.4f}")
                if analysis.take_profit:
                    st.success(f"💰 **Take Profit:** ${analysis.take_profit:,.4f}")
            
            with col2:
                st.subheader("⚖️ Position Management")
                if analysis.max_position_size_usd:
                    st.metric("Recommended Position", f"${analysis.max_position_size_usd:,.0f}")
                if analysis.risk_percentage:
                    st.metric("Portfolio Risk", f"{analysis.risk_percentage:.1f}%")
                
                # Risk/reward ratio
                if analysis.buy_price and analysis.take_profit and analysis.stop_loss:
                    potential_gain = analysis.take_profit - analysis.buy_price
                    potential_loss = analysis.buy_price - analysis.stop_loss
                    if potential_loss > 0:
                        risk_reward = potential_gain / potential_loss
                        st.metric("Risk/Reward Ratio", f"1:{risk_reward:.2f}")
            
            # Analysis details
            tab1, tab2, tab3 = st.tabs(["🔬 Technical Analysis", "📰 Fundamental Analysis", "⚠️ Risk Factors"])
            
            with tab1:
                st.markdown(f"**Technical Analysis:** {analysis.technical_analysis}")
            
            with tab2:
                st.markdown(f"**Fundamental Analysis:** {analysis.fundamental_analysis}")
            
            with tab3:
                st.markdown("**Risk Factors:**")
                for risk in analysis.risk_factors:
                    st.warning(f"• {risk}")
            
            # Action buttons
            st.markdown("---")
            st.subheader("🚀 Take Action")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💬 Discuss with AI", use_container_width=True):
                    st.session_state.page = "chat"
                    # Add analysis context to chat
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    analysis_summary = f"I just analyzed {crypto_name} and got a {analysis.recommendation} recommendation with {analysis.confidence_score}/10 confidence. Can you help me understand this better and what I should do next?"
                    st.session_state.chat_history.append({
                        'question': analysis_summary,
                        'response': f"I see you analyzed {crypto_name}! Based on the {analysis.recommendation} recommendation with {analysis.confidence_score}/10 confidence, here's what you should know:\n\n**Key Points:**\n- Current Price: ${analysis.current_price:,.4f}\n- Recommendation: {analysis.recommendation}\n- Time Horizon: {analysis.time_horizon}\n\n**Technical Summary:** {analysis.technical_analysis[:150]}...\n\nWhat specific aspect would you like me to explain further?",
                        'actions': ["Explain the recommendation reasoning", "Discuss position sizing", "Compare with portfolio holdings"]
                    })
                    st.rerun()
            
            with col2:
                if analysis.recommendation in ["STRONG_BUY", "BUY"] and st.button("📊 Add to Watchlist", use_container_width=True):
                    if 'watchlist' not in st.session_state:
                        st.session_state.watchlist = []
                    if selected_crypto not in st.session_state.watchlist:
                        st.session_state.watchlist.append(selected_crypto)
                        st.success(f"Added {crypto_name} to watchlist!")
                    else:
                        st.info(f"{crypto_name} already in watchlist")
            
            with col3:
                if analysis.buy_price and st.button("🔗 Create Binance Alert", use_container_width=True):
                    binance_url = f"https://www.binance.com/en/trade/{selected_crypto.upper().replace('-', '')}_USDT"
                    st.markdown(f"[Open {crypto_name} on Binance]({binance_url})")
        
        # Watchlist section
        if 'watchlist' in st.session_state and st.session_state.watchlist:
            st.markdown("---")
            st.subheader("👁️ Your Watchlist")
            
            watchlist_cols = st.columns(min(len(st.session_state.watchlist), 4))
            for i, crypto in enumerate(st.session_state.watchlist):
                with watchlist_cols[i % 4]:
                    crypto_display = crypto.replace('-', ' ').title()
                    if st.button(f"📊 {crypto_display}", key=f"watchlist_{i}", use_container_width=True):
                        # Quick analysis of watchlist item
                        with st.spinner(f"Quick analysis of {crypto_display}..."):
                            quick_analysis = get_crypto_price_analysis(crypto, risk_profile, portfolio_size)
                            if quick_analysis:
                                st.session_state.current_analysis = quick_analysis
                                st.session_state.analyzed_crypto = crypto
                                st.rerun()
                    st.error(f"🎯 **Sell Price:** ${analysis.sell_price:,.4f}")
                if analysis.support_level:
                    st.info(f"🛡️ **Support:** ${analysis.support_level:,.4f}")
                if analysis.resistance_level:
                    st.warning(f"⚠️ **Resistance:** ${analysis.resistance_level:,.4f}")
            
            with col2:
                st.subheader("🎯 Risk Management")
                if analysis.stop_loss:
                    st.error(f"🛑 **Stop Loss:** ${analysis.stop_loss:,.4f}")
                if analysis.take_profit:
                    st.success(f"💰 **Take Profit:** ${analysis.take_profit:,.4f}")
                
                # Calculate potential returns
                if analysis.buy_price and analysis.take_profit:
                    potential_return = ((analysis.take_profit - analysis.buy_price) / analysis.buy_price) * 100
                    st.metric("Potential Return", f"{potential_return:+.1f}%")
                
                if analysis.buy_price and analysis.stop_loss:
                    max_loss = ((analysis.stop_loss - analysis.buy_price) / analysis.buy_price) * 100
                    st.metric("Max Loss", f"{max_loss:.1f}%")
            
            # Analysis details
            tab1, tab2, tab3 = st.tabs(["🔬 Technical Analysis", "📰 Fundamental Analysis", "⚠️ Risk Factors"])
            
            with tab1:
                st.markdown(analysis.technical_analysis)
            
            with tab2:
                st.markdown(analysis.fundamental_analysis)
            
            with tab3:
                st.markdown("**Key Risks to Consider:**")
                for i, risk in enumerate(analysis.risk_factors, 1):
                    st.warning(f"{i}. {risk}")
            
            # Action buttons
            st.markdown("---")
            st.subheader("🚀 Take Action")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if analysis.buy_price:
                    binance_url = f"https://www.binance.com/en/trade/{analysis.symbol.upper()}_USDT"
                    st.link_button("📈 Trade on Binance", binance_url, use_container_width=True)
            
            with col2:
                if st.button("🔄 Refresh Analysis", use_container_width=True):
                    # Clear cache and rerun analysis
                    get_crypto_technical_analysis.clear()
                    if 'current_analysis' in st.session_state:
                        del st.session_state.current_analysis
                    st.rerun()
            
            with col3:
                if st.button("📊 Add to Watchlist", use_container_width=True):
                    if 'watchlist' not in st.session_state:
                        st.session_state.watchlist = []
                    if selected_crypto not in st.session_state.watchlist:
                        st.session_state.watchlist.append(selected_crypto)
                        st.success(f"✅ Added {crypto_name} to watchlist!")
                    else:
                        st.info(f"ℹ️ {crypto_name} is already in your watchlist")
        
        # Watchlist section
        if 'watchlist' in st.session_state and st.session_state.watchlist:
            st.markdown("---")
            st.subheader("👁️ Your Watchlist")
            
            watchlist_cols = st.columns(min(len(st.session_state.watchlist), 4))
            for i, crypto in enumerate(st.session_state.watchlist):
                with watchlist_cols[i % 4]:
                    if st.button(f"📊 {crypto.replace('-', ' ').title()}", key=f"watchlist_{crypto}"):
                        st.session_state.page = "price_analysis"
                        # Set the selected crypto and analyze
                        with st.spinner(f"Analyzing {crypto}..."):
                            analysis = get_crypto_price_analysis(crypto, risk_profile, portfolio_size)
                            if analysis:
                                st.session_state.current_analysis = analysis
                                st.session_state.analyzed_crypto = crypto
                                st.rerun()

elif st.session_state.page == "screener":
    ### MARKET SCREENER PAGE ###
    st.header("🔍 Crypto Market Screener")
    st.markdown("Discover investment opportunities with advanced filtering and AI-powered price analysis.")
    
    # Screener Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sort_options = {
            "Market Cap": "market_cap_desc",
            "Price": "price_desc", 
            "24h Change": "price_change_percentage_24h_desc",
            "7d Change": "price_change_percentage_7d_desc",
            "Volume": "volume_desc"
        }
        sort_by = st.selectbox("Sort By", list(sort_options.keys()))
    
    with col2:
        min_market_cap = st.number_input("Min Market Cap ($M)", min_value=0, value=100, step=50) * 1000000
    
    with col3:
        categories = ["", "defi", "smart-contract-platform", "meme-token", "layer-1", "layer-2"]
        category = st.selectbox("Category", categories)
    
    with col4:
        show_analysis = st.checkbox("🧠 Show AI Analysis", help="Get AI buy/sell recommendations for each coin")
    
    # Get screener data
    screener_df = get_crypto_screener(sort_options[sort_by], category, min_market_cap)
    
    # Debug: Show available columns (remove this after testing)
    if not screener_df.empty and st.checkbox("🔧 Debug: Show available columns", value=False):
        st.write("Available columns:", screener_df.columns.tolist())
    
    if not screener_df.empty:
        st.subheader(f"📊 Found {len(screener_df)} cryptocurrencies matching your criteria")
        
        # Display screener results
        for index, row in screener_df.head(10).iterrows():  # Show top 10 with detailed analysis
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    # Basic info
                    change_color = "green" if row['price_change_percentage_24h'] > 0 else "red"
                    st.markdown(f"### 🪙 {row['name']} ({row['symbol'].upper()})")
                    st.markdown(f"**Price:** ${row['current_price']:,.4f}")
                    st.markdown(f"**24h Change:** :{change_color}[{row['price_change_percentage_24h']:+.2f}%]")
                    st.markdown(f"**Market Cap:** ${row['market_cap']:,.0f}")
                    st.markdown(f"**Volume:** ${row['total_volume']:,.0f}")
                
                with col2:
                    if show_analysis and st.session_state.get('openai_api_key'):
                        # Get quick AI analysis
                        if st.button(f"🔬 Analyze {row['symbol'].upper()}", key=f"analyze_{row['id']}", use_container_width=True):
                            with st.spinner(f"Analyzing {row['name']}..."):
                                try:
                                    analysis = get_crypto_price_analysis(row['id'], st.session_state.risk_profile, 10000)
                                    if analysis:
                                        # Display quick recommendation
                                        rec_color = "green" if analysis.recommendation in ["STRONG_BUY", "BUY"] else "red" if analysis.recommendation in ["SELL", "STRONG_SELL"] else "orange"
                                        st.markdown(f"**Recommendation:** :{rec_color}[{analysis.recommendation}]")
                                        st.markdown(f"**Confidence:** {analysis.confidence_score}/10")
                                        if analysis.buy_price:
                                            st.markdown(f"**Buy Price:** ${analysis.buy_price:,.4f}")
                                        if analysis.sell_price:
                                            st.markdown(f"**Sell Price:** ${analysis.sell_price:,.4f}")
                                        
                                        # Store analysis for detailed view
                                        st.session_state[f"analysis_{row['id']}"] = analysis
                                except Exception as e:
                                    st.error(f"Analysis failed: {e}")
                        
                        # Show stored analysis if available
                        if f"analysis_{row['id']}" in st.session_state:
                            analysis = st.session_state[f"analysis_{row['id']}"]
                            rec_color = "green" if analysis.recommendation in ["STRONG_BUY", "BUY"] else "red" if analysis.recommendation in ["SELL", "STRONG_SELL"] else "orange"
                            st.markdown(f"**🎯 {analysis.recommendation}** ({analysis.confidence_score}/10)")
                            if analysis.buy_price:
                                st.success(f"Buy: ${analysis.buy_price:,.4f}")
                            if analysis.sell_price:
                                st.error(f"Sell: ${analysis.sell_price:,.4f}")
                    else:
                        if not st.session_state.get('openai_api_key'):
                            st.info("💡 Enter OpenAI API key to get AI analysis")
                        else:
                            st.markdown("**Quick Stats:**")
                            # Check for different possible 7d change column names
                            seven_day_change = None
                            if row.get('price_change_percentage_7d_in_currency') is not None:
                                seven_day_change = row['price_change_percentage_7d_in_currency']
                            elif row.get('price_change_percentage_7d') is not None:
                                seven_day_change = row['price_change_percentage_7d']
                            
                            if seven_day_change is not None:
                                change_7d_color = "green" if seven_day_change > 0 else "red"
                                st.markdown(f"7d: :{change_7d_color}[{seven_day_change:+.2f}%]")
                            if row.get('market_cap_rank'):
                                st.markdown(f"Rank: #{row['market_cap_rank']}")
                
                with col3:
                    # Action buttons
                    binance_url = f"https://www.binance.com/en/trade/{row['symbol'].upper()}_USDT"
                    st.link_button("📈 Trade", binance_url, use_container_width=True)
                    
                    if st.button("📊 Full Analysis", key=f"full_analysis_{row['id']}", use_container_width=True):
                        # Navigate to price analysis page with this crypto
                        st.session_state.page = "price_analysis"
                        st.session_state.selected_crypto_for_analysis = row['id']
                        st.rerun()
                    
                    if st.button("➕ Watchlist", key=f"watchlist_{row['id']}", use_container_width=True):
                        if 'watchlist' not in st.session_state:
                            st.session_state.watchlist = []
                        if row['id'] not in st.session_state.watchlist:
                            st.session_state.watchlist.append(row['id'])
                            st.success("✅ Added!")
                        else:
                            st.info("Already added!")
        
        # Show remaining results in table format
        if len(screener_df) > 10:
            st.markdown("---")
            st.subheader(f"📋 Remaining {len(screener_df) - 10} coins (Table View)")
            
            # Check available columns and use the correct ones
            available_cols = screener_df.columns.tolist()
            display_cols = ['name', 'symbol', 'current_price', 'market_cap', 'price_change_percentage_24h', 'total_volume']
            
            # Add 7d change column if available (different possible names)
            if 'price_change_percentage_7d_in_currency' in available_cols:
                display_cols.insert(-1, 'price_change_percentage_7d_in_currency')
                seven_day_col = 'price_change_percentage_7d_in_currency'
            elif 'price_change_percentage_7d' in available_cols:
                display_cols.insert(-1, 'price_change_percentage_7d')
                seven_day_col = 'price_change_percentage_7d'
            else:
                seven_day_col = None
            
            # Only select columns that exist
            existing_cols = [col for col in display_cols if col in available_cols]
            remaining_df = screener_df.iloc[10:][existing_cols].copy()
            
            # Format the data
            remaining_df['current_price'] = remaining_df['current_price'].apply(lambda x: f"${x:,.4f}")
            remaining_df['market_cap'] = remaining_df['market_cap'].apply(lambda x: f"${x:,.0f}")
            remaining_df['total_volume'] = remaining_df['total_volume'].apply(lambda x: f"${x:,.0f}")
            remaining_df['price_change_percentage_24h'] = remaining_df['price_change_percentage_24h'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
            
            if seven_day_col and seven_day_col in remaining_df.columns:
                remaining_df[seven_day_col] = remaining_df[seven_day_col].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
            
            # Set column names for display
            column_names = ['Name', 'Symbol', 'Price', 'Market Cap', '24h Change']
            if seven_day_col and seven_day_col in remaining_df.columns:
                column_names.append('7d Change')
            column_names.append('Volume')
            
            remaining_df.columns = column_names[:len(remaining_df.columns)]
            
            st.dataframe(remaining_df, use_container_width=True, height=400)
    
    else:
        st.warning("No data available for the selected criteria.")
        st.info("💡 Try adjusting your filters or check back later for updated market data.")
        
        # Suggest some popular cryptos
        st.markdown("**Popular cryptos you might want to analyze:**")
        popular_suggestions = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
        suggest_cols = st.columns(len(popular_suggestions))
        for i, crypto in enumerate(popular_suggestions):
            with suggest_cols[i]:
                if st.button(f"📊 {crypto.title()}", key=f"suggest_{crypto}"):
                    st.session_state.page = "price_analysis"
                    st.session_state.selected_crypto_for_analysis = crypto
                    st.rerun()
    
    # Quick access to popular cryptos
    st.markdown("---")
    st.subheader("🔥 Quick Analysis - Popular Cryptos")
    
    popular_cryptos = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 'polkadot', 'dogecoin', 'avalanche-2']
    popular_cols = st.columns(4)
    
    for i, crypto in enumerate(popular_cryptos):
        with popular_cols[i % 4]:
            if st.button(f"📊 {crypto.replace('-', ' ').title()}", key=f"popular_{crypto}", use_container_width=True):
                st.session_state.page = "price_analysis"
                st.session_state.selected_crypto_for_analysis = crypto
                st.rerun()

elif st.session_state.page == "smart_research":
    ### SMART INVESTMENT RESEARCH PAGE ###
    st.header("🧠 Smart Investment Research")
    st.markdown("AI-powered investment research using real-time internet data to find the best crypto opportunities.")
    
    if not st.session_state.get('openai_api_key'):
        st.warning("⚠️ Please enter your OpenAI API key to use Smart Research.")
    elif not st.session_state.get('tavily_api_key'):
        st.warning("⚠️ Please enter your Tavily API key to access internet research.")
    else:
        
        # Research controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            research_risk_profile = st.selectbox(
                "🎯 Research Focus", 
                CONFIG["risk_profiles"],
                index=1,
                help="Tailor research to your risk appetite"
            )
        
        with col2:
            research_timeframe = st.selectbox(
                "⏰ Investment Timeframe",
                ["Short-term (1-4 weeks)", "Medium-term (1-6 months)", "Long-term (6+ months)"],
                index=1
            )
        
        with col3:
            portfolio_budget = st.number_input(
                "💰 Research Budget ($)",
                min_value=100,
                value=5000,
                step=500,
                help="Budget for new investments to tailor recommendations"
            )
        
        st.markdown("---")
        
        # Research execution
        if st.button("🔬 Conduct Smart Research", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is conducting comprehensive market research..."):
                research_report, research_data, search_queries = get_smart_investment_recommendations(research_risk_profile)
                
                if research_report and research_data and search_queries:
                    st.session_state.research_report = research_report
                    st.session_state.research_data = research_data
                    st.session_state.search_queries = search_queries
                    st.session_state.research_timestamp = datetime.now()
                    st.success("✅ Research completed!")
                else:
                    st.error("❌ Research failed. Please check your API keys and try again.")
        
        # Display research results
        if 'research_report' in st.session_state:
            report = st.session_state.research_report
            timestamp = st.session_state.research_timestamp.strftime("%B %d, %Y at %H:%M")
            
            st.markdown("---")
            st.subheader(f"📋 Research Report - {timestamp}")
            
            # Market Overview
            with st.container(border=True):
                st.markdown("### 🌍 Market Overview")
                st.write(report.market_overview)
                
                # Key sectors
                if report.sectors_to_watch:
                    st.markdown("**🔥 Hot Sectors:**")
                    sectors_text = " • ".join([f"**{sector}**" for sector in report.sectors_to_watch])
                    st.markdown(sectors_text)
            
            # Top Investment Opportunities
            st.markdown("### 💎 Top Investment Opportunities")
            
            if report.top_opportunities:
                for i, opportunity in enumerate(report.top_opportunities, 1):
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            # Recommendation badge
                            rec_color = {"STRONG_BUY": "🟢", "BUY": "🟢", "HOLD": "🟡", "AVOID": "🔴"}
                            st.markdown(f"## {rec_color.get(opportunity.recommendation, '⚪')} {opportunity.crypto_symbol}")
                            st.markdown(f"**{opportunity.recommendation}** (Confidence: {opportunity.confidence_score}/10)")
                            
                            # Investment thesis
                            st.markdown("**💡 Investment Thesis:**")
                            st.write(opportunity.investment_thesis)
                        
                        with col2:
                            st.markdown("**🚀 Catalysts:**")
                            for catalyst in opportunity.catalysts[:3]:  # Show top 3
                                st.markdown(f"• {catalyst}")
                        
                        with col3:
                            st.markdown("**⚠️ Risks:**")
                            for risk in opportunity.risks[:3]:  # Show top 3
                                st.markdown(f"• {risk}")
                        
                        # Additional details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if opportunity.price_target:
                                st.metric("Price Target", f"${opportunity.price_target:,.4f}")
                        with col2:
                            st.metric("Time Horizon", opportunity.time_horizon)
                        with col3:
                            # Calculate suggested position size
                            position_multiplier = {"STRONG_BUY": 0.15, "BUY": 0.10, "HOLD": 0.05, "AVOID": 0.0}
                            suggested_amount = portfolio_budget * position_multiplier.get(opportunity.recommendation, 0.05)
                            if suggested_amount > 0:
                                st.metric("Suggested Position", f"${suggested_amount:,.0f}")
                        
                        # Action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            # Try to create trading link
                            crypto_symbol = opportunity.crypto_symbol.upper()
                            binance_url = f"https://www.binance.com/en/trade/{crypto_symbol}_USDT"
                            st.link_button(f"📈 Trade {crypto_symbol}", binance_url, use_container_width=True)
                        
                        with col2:
                            if st.button(f"📊 Deep Analysis", key=f"analyze_{opportunity.crypto_symbol}", use_container_width=True):
                                # Navigate to price analysis for this crypto
                                crypto_id = COIN_ID_MAP.get(crypto_symbol.upper())
                                if crypto_id:
                                    st.session_state.page = "price_analysis"
                                    st.session_state.selected_crypto_for_analysis = crypto_id
                                    st.rerun()
            
            # Market Risks
            if report.market_risks:
                st.markdown("### ⚠️ Market Risks to Monitor")
                risk_cols = st.columns(2)
                for i, risk in enumerate(report.market_risks):
                    with risk_cols[i % 2]:
                        st.warning(f"• {risk}")
            
            # Research Summary
            with st.container(border=True):
                st.markdown("### 📊 Executive Summary")
                st.write(report.research_summary)
            
            # Search Results Section - NEW
            if 'research_data' in st.session_state and 'search_queries' in st.session_state:
                st.markdown("---")
                
                # Toggle to show/hide search details
                show_search_details = st.checkbox("🔍 Show Internet Search Details", 
                                                 value=False, 
                                                 help="View the search queries and results that powered this research")
                
                if show_search_details:
                    search_queries = st.session_state.search_queries
                    research_data = st.session_state.research_data
                    
                    st.subheader("🔍 Internet Search Results")
                    st.markdown("*This shows the actual search queries and results that were used to generate the AI recommendations above.*")
                    
                    if not research_data:
                        st.warning("No research data found. The search might have failed.")
                        st.stop()
                    
                    # Group research data by query for better organization
                    query_results = {}
                    for item in research_data:
                        query = item['query']
                        if query not in query_results:
                            query_results[query] = []
                        query_results[query].append(item)
                    
                    # Display search queries by category
                    search_categories = [
                        ("🔥 Trending Opportunities", search_queries.trending_opportunities),
                        ("🏢 Sector Analysis", search_queries.sector_analysis),
                        ("📰 Market Events", search_queries.market_events),
                        ("📈 Technical Signals", search_queries.technical_signals),
                        ("⚡ Fundamental Catalysts", search_queries.fundamental_catalysts)
                    ]
                    
                    for category_name, queries in search_categories:
                        if queries:  # Only show categories that have queries
                            with st.expander(f"{category_name} ({len(queries)} searches)", expanded=False):
                                for i, query in enumerate(queries, 1):
                                    st.markdown(f"**Search {i}:** {query}")
                                    
                                    # Show results for this query
                                    if query in query_results:
                                        results = query_results[query]
                                        st.markdown(f"*Found {len(results)} results:*")
                                        
                                        for j, result in enumerate(results, 1):
                                            with st.container(border=True):
                                                col1, col2 = st.columns([3, 1])
                                                
                                                with col1:
                                                    st.markdown(f"**{j}. {result['title']}**")
                                                    # Show first 200 characters of content
                                                    content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                                                    st.markdown(f"*{content_preview}*")
                                                    
                                                    if result.get('published_date'):
                                                        st.caption(f"📅 Published: {result['published_date']}")
                                                
                                                with col2:
                                                    if result['url']:
                                                        st.link_button("🔗 Read Full Article", result['url'], use_container_width=True)
                                    else:
                                        st.info("No results found for this query")
                                    
                                    st.markdown("---")
                    
                    # Summary stats
                    total_results = len(research_data)
                    unique_sources = len(set(item['url'] for item in research_data if item['url']))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Search Results", total_results)
                    with col2:
                        st.metric("Unique Sources", unique_sources)
                    with col3:
                        st.metric("Search Categories", len([cat for cat, queries in search_categories if queries]))
            else:
                st.info("🔍 Search results will appear here after conducting Smart Research.")
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Research", use_container_width=True):
                    # Clear cache and rerun research
                    get_smart_investment_recommendations.clear()
                    if 'research_report' in st.session_state:
                        del st.session_state.research_report
                    if 'research_data' in st.session_state:
                        del st.session_state.research_data
                    if 'search_queries' in st.session_state:
                        del st.session_state.search_queries
                    st.rerun()
            
            with col2:
                # Download research report
                report_data = {
                    "timestamp": timestamp,
                    "market_overview": report.market_overview,
                    "opportunities": [opp.model_dump() for opp in report.top_opportunities],
                    "sectors": report.sectors_to_watch,
                    "risks": report.market_risks,
                    "summary": report.research_summary,
                    "search_queries": st.session_state.search_queries.model_dump() if 'search_queries' in st.session_state else None,
                    "search_results": st.session_state.research_data if 'research_data' in st.session_state else None
                }
                st.download_button(
                    "📄 Download Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"smart_research_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                if st.button("💬 Discuss with AI", use_container_width=True):
                    st.session_state.page = "chat"
                    # Add research context to chat
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({
                        'question': f"I just completed smart research. Can you help me understand the findings and next steps? Research completed on {timestamp}.",
                        'response': f"I see you've completed smart research! Based on the report from {timestamp}, here are the key findings:\n\n**Market Overview:** {report.market_overview[:200]}...\n\n**Top Opportunities:** {len(report.top_opportunities)} investments identified.\n\nWhat specific aspect would you like to discuss further?",
                        'actions': ["Ask about specific recommendations", "Discuss portfolio allocation", "Understand market risks"]
                    })
                    st.rerun()
        
        # Quick research shortcuts
        st.markdown("---")
        st.subheader("⚡ Quick Research")
        st.markdown("Pre-configured research for specific themes:")
        
        quick_research_options = {
            "🤖 AI & Blockchain": "AI cryptocurrency tokens and blockchain artificial intelligence projects January 2025",
            "🎮 Gaming & Metaverse": "GameFi metaverse cryptocurrency gaming tokens January 2025 trends",
            "🏦 DeFi Innovation": "DeFi decentralized finance new protocols yield farming January 2025",
            "⚡ Layer 2 Solutions": "Layer 2 scaling solutions Ethereum Polygon Arbitrum January 2025",
            "🌱 Green Crypto": "Sustainable eco-friendly green cryptocurrency carbon neutral January 2025",
            "🏛️ Institutional Adoption": "institutional cryptocurrency adoption corporate Bitcoin Ethereum January 2025"
        }
        
        quick_cols = st.columns(3)
        for i, (theme, query) in enumerate(quick_research_options.items()):
            with quick_cols[i % 3]:
                if st.button(theme, use_container_width=True, key=f"quick_{i}"):
                    with st.spinner(f"Researching {theme}..."):
                        # Perform targeted research
                        tavily_client = st.session_state.get('tavily_client')
                        if not tavily_client:
                            st.error("Tavily API key required for quick research")
                            st.stop()
                        try:
                            search_results = tavily_client.search(
                                query=query,
                                search_depth="advanced",
                                max_results=5
                            )
                            
                            context = " ".join([res['content'] for res in search_results['results']])
                            
                            quick_prompt = f"""
                            Based on the research about {theme}, provide a brief analysis of investment opportunities.
                            
                            Research data: {context[:2000]}...
                            
                            Provide: 1) Key findings, 2) Top 2-3 specific cryptocurrencies to watch, 3) Main risks
                            """
                            
                            quick_analysis = get_llm_response(quick_prompt, st.session_state.openai_api_key, ChatResponse)
                            if quick_analysis:
                                st.success(f"**{theme} Research:**")
                                st.write(quick_analysis.response)
                        except Exception as e:
                            st.error(f"Quick research failed: {e}")

elif st.session_state.page == "chat":
    ### AI CHAT ASSISTANT PAGE ###
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("💬 AI Crypto Assistant")
        st.markdown("Ask me anything about cryptocurrency, trading strategies, or your portfolio!")
    with col2:
        if st.button("🗑️ Clear Chat", help="Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
    
    if not st.session_state.get('openai_api_key'):
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the AI assistant.")
        st.info("💡 The AI assistant can help you with:\n- Cryptocurrency analysis and advice\n- Trading strategies and recommendations\n- Portfolio analysis and suggestions\n- Market insights and explanations")
    else:
        # Show portfolio context if available
        if 'portfolio_summary' in st.session_state:
            portfolio_df = st.session_state.portfolio_summary['dataframe']
            total_value = portfolio_df['USD Value'].sum()
            fetched_at = st.session_state.portfolio_summary.get('fetched_at', 'Unknown')
            
            with st.expander("📊 Current Portfolio Context", expanded=False):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Value", f"${total_value:,.2f}")
                with col2:
                    st.metric("Assets", f"{len(portfolio_df)}")
                with col3:
                    portfolio_24h = (portfolio_df['USD Value'] * portfolio_df['24h Perf (%)'] / 100).sum() / total_value * 100
                    st.metric("24h Change", f"{portfolio_24h:+.2f}%")
                with col4:
                    risk_profile = st.session_state.get('risk_profile', 'Balanced')
                    st.metric("Risk Profile", risk_profile)
                with col5:
                    if st.button("🔄 Refresh Portfolio", help="Update portfolio data for latest market prices"):
                        # Clear portfolio cache to force refresh
                        if hasattr(st.session_state, 'portfolio_summary'):
                            del st.session_state.portfolio_summary
                        st.info("Portfolio data cleared. Use 'Analyze Portfolio' in sidebar to refresh.")
                
                st.caption(f"� Portfolio data fetched: {fetched_at}")
                st.info("�💡 The AI assistant has full access to your portfolio data and will provide personalized advice based on your actual holdings, performance, and risk profile.")
        else:
            with st.container(border=True):
                st.warning("� **No Portfolio Data Available**")
                st.markdown("Connect your Binance account to get personalized advice!")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("✅ **With Portfolio Connection:**")
                    st.markdown("• Personalized recommendations based on your holdings")
                    st.markdown("• Risk analysis for your specific assets")
                    st.markdown("• Performance insights and optimization suggestions")
                    st.markdown("• Rebalancing recommendations")
                with col2:
                    st.markdown("⚠️ **Without Portfolio:**")
                    st.markdown("• Generic cryptocurrency advice only")
                    st.markdown("• No personalized risk assessment")
                    st.markdown("• Limited context for recommendations")
                    
                if st.button("🚀 Connect Portfolio Now", use_container_width=True, type="primary"):
                    st.session_state.page = "portfolio"
                    st.rerun()
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            # Add personalized welcome message
            portfolio_status = "with your portfolio data" if 'portfolio_summary' in st.session_state else "without portfolio data (connect for personalized advice)"
            welcome_message = f"👋 Hello! I'm your AI crypto assistant {portfolio_status}. I can help you with cryptocurrency analysis, trading strategies, portfolio advice, and market insights. What would you like to know?"
            
            welcome_actions = [
                "Ask about a specific cryptocurrency",
                "Get trading recommendations", 
                "Analyze market trends"
            ]
            
            if 'portfolio_summary' in st.session_state:
                welcome_actions.extend([
                    "Review my portfolio performance",
                    "Suggest portfolio rebalancing",
                    "Identify risk in my holdings"
                ])
            else:
                welcome_actions.append("Connect portfolio for personalized advice")
            
            st.session_state.chat_history.append({
                'question': None,
                'response': welcome_message,
                'actions': welcome_actions,
                'is_welcome': True
            })
        
        # Create a container for the chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                # Only show user message if it's not a welcome message
                if not chat.get('is_welcome', False):
                    with st.chat_message("user"):
                        st.write(chat['question'])
                
                # Show assistant response
                with st.chat_message("assistant"):
                    st.write(chat['response'])
                    if chat.get('actions'):
                        st.markdown("**💡 Suggested Actions:**")
                        for action in chat['actions']:
                            st.markdown(f"• {action}")
        
        # Chat input at the bottom
        user_question = st.chat_input("Ask your crypto question...")
        
        if user_question:
            # Show user message immediately
            with st.chat_message("user"):
                st.write(user_question)
            
            # Show AI thinking and response
            with st.chat_message("assistant"):
                with st.spinner("🤔 AI is analyzing your question with portfolio context..."):
                    try:
                        # Get portfolio context if available
                        portfolio_context = None
                        if 'portfolio_summary' in st.session_state:
                            portfolio_context = st.session_state.portfolio_summary['dataframe']
                        
                        # Check API key
                        if not st.session_state.get('openai_api_key'):
                            st.error("❌ OpenAI API key is missing. Please enter it in the sidebar.")
                            st.stop()
                        
                        # Show debug info
                        st.caption(f"🔍 Processing question: {user_question[:50]}...")
                        
                        # Get AI response
                        response = get_ai_chat_response(user_question, portfolio_context)
                        
                        # Clear debug info
                        st.empty()
                        
                    except Exception as e:
                        st.error(f"❌ Error during AI processing: {str(e)}")
                        print(f"Debug - Chat error: {e}")
                        response = None
                
                if response:
                    # Display response immediately
                    st.write(response.response)
                    if response.suggested_actions:
                        st.markdown("**💡 Suggested Actions:**")
                        for action in response.suggested_actions:
                            st.markdown(f"• {action}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'response': response.response,
                        'actions': response.suggested_actions
                    })
                else:
                    # Enhanced fallback with more helpful information
                    st.error("❌ Sorry, I couldn't process your question.")
                    
                    # Try to provide a basic response without AI
                    fallback_response = get_fallback_response(user_question, portfolio_context)
                    if fallback_response:
                        st.info("💡 Here's a basic response while we troubleshoot the AI connection:")
                        st.write(fallback_response)
                    else:
                        st.write("Please check your OpenAI API key and try again. You can also try:")
                        st.markdown("• Asking a simpler question")
                        st.markdown("• Refreshing the page")
                        st.markdown("• Checking your internet connection")
                    
                    # Add fallback to chat history
                    fallback_msg = fallback_response if fallback_response else "I apologize, but I'm having trouble processing your request right now. Please make sure your OpenAI API key is configured correctly in the sidebar and try again."
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'response': fallback_msg,
                        'actions': ["Check your OpenAI API key in the sidebar", "Try asking a simpler question", "Refresh the page and try again"]
                    })
        
        # Add some helpful examples based on portfolio status
        if len(st.session_state.chat_history) <= 1:
            st.markdown("---")
            
            if 'portfolio_summary' in st.session_state:
                st.subheader("💡 Portfolio-Specific Questions You Can Ask:")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("� How is my portfolio performing?", use_container_width=True):
                        st.session_state.example_question = "How is my portfolio performing and what should I adjust?"
                        st.rerun()
                    
                    if st.button("⚖️ Should I rebalance my portfolio?", use_container_width=True):
                        st.session_state.example_question = "Should I rebalance my portfolio based on current allocations and performance?"
                        st.rerun()
                    
                    if st.button("🎯 Which of my coins should I sell?", use_container_width=True):
                        st.session_state.example_question = "Which of my current holdings should I consider selling and why?"
                        st.rerun()
                
                with col2:
                    if st.button("� What new coins should I buy?", use_container_width=True):
                        st.session_state.example_question = "Based on my current portfolio, what new cryptocurrencies should I consider buying?"
                        st.rerun()
                    
                    if st.button("⚠️ What are my portfolio risks?", use_container_width=True):
                        st.session_state.example_question = "What are the main risks in my current portfolio and how can I mitigate them?"
                        st.rerun()
                    
                    if st.button("💰 Should I take profits now?", use_container_width=True):
                        st.session_state.example_question = "Based on my portfolio performance, should I take profits on any of my holdings?"
                        st.rerun()
            else:
                st.subheader("💡 General Crypto Questions You Can Ask:")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📈 What's the current Bitcoin trend?", use_container_width=True):
                        st.session_state.example_question = "What's the current Bitcoin trend and should I buy now?"
                        st.rerun()
                    
                    if st.button("🎯 Best altcoins for 2025?", use_container_width=True):
                        st.session_state.example_question = "What are the best altcoins to invest in for 2025?"
                        st.rerun()
                
                with col2:
                    if st.button("💼 How to build a crypto portfolio?", use_container_width=True):
                        st.session_state.example_question = "How should I build a diversified cryptocurrency portfolio?"
                        st.rerun()
                    
                    if st.button("📊 Explain technical analysis", use_container_width=True):
                        st.session_state.example_question = "Can you explain how to use RSI and moving averages for crypto trading?"
                        st.rerun()
        
        # Handle example questions
        if 'example_question' in st.session_state:
            example_q = st.session_state.example_question
            del st.session_state.example_question
            
            # Process the example question
            with st.chat_message("user"):
                st.write(example_q)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 AI is analyzing your question with portfolio context..."):
                    portfolio_context = None
                    if 'portfolio_summary' in st.session_state:
                        portfolio_context = st.session_state.portfolio_summary['dataframe']
                    
                    response = get_ai_chat_response(example_q, portfolio_context)
                
                if response:
                    st.write(response.response)
                    if response.suggested_actions:
                        st.markdown("**💡 Suggested Actions:**")
                        for action in response.suggested_actions:
                            st.markdown(f"• {action}")
                    
                    st.session_state.chat_history.append({
                        'question': example_q,
                        'response': response.response,
                        'actions': response.suggested_actions
                    })

elif st.session_state.page == "learn":
    ### EDUCATIONAL CONTENT PAGE ###
    st.header("📚 Crypto Learning Center")
    
    tab1, tab2, tab3 = st.tabs(["📖 Basics", "📊 Technical Analysis", "💡 Strategies"])
    
    with tab1:
        st.subheader("Cryptocurrency Basics")
        
        with st.expander("What is Cryptocurrency?"):
            st.markdown("""
            Cryptocurrency is a digital or virtual currency secured by cryptography, making it nearly impossible to counterfeit. 
            Key characteristics:
            - **Decentralized**: Not controlled by any government or financial institution
            - **Blockchain-based**: Transactions recorded on a distributed ledger
            - **Limited supply**: Most cryptocurrencies have a fixed maximum supply
            - **Pseudonymous**: Transactions are recorded but user identities are masked
            """)
        
        with st.expander("Types of Cryptocurrencies"):
            st.markdown("""
            - **Bitcoin (BTC)**: The first and largest cryptocurrency, digital gold
            - **Ethereum (ETH)**: Smart contract platform, powers DeFi and NFTs
            - **Stablecoins**: Pegged to fiat currencies (USDT, USDC)
            - **Altcoins**: All cryptocurrencies other than Bitcoin
            - **DeFi Tokens**: Power decentralized finance protocols
            - **Meme Coins**: Community-driven tokens (DOGE, SHIB)
            """)
    
    with tab2:
        st.subheader("Technical Analysis Fundamentals")
        
        with st.expander("Key Indicators"):
            st.markdown("""
            - **RSI (Relative Strength Index)**: Measures overbought/oversold conditions (0-100)
              - RSI > 70: Potentially overbought
              - RSI < 30: Potentially oversold
            
            - **Moving Averages**: Smooth out price action
              - SMA 50: Short-term trend
              - SMA 200: Long-term trend
              - Golden Cross: 50 SMA crosses above 200 SMA (bullish)
              - Death Cross: 50 SMA crosses below 200 SMA (bearish)
            
            - **Support & Resistance**: Key price levels where buying/selling pressure emerges
            """)
        
        with st.expander("Chart Patterns"):
            st.markdown("""
            - **Trend Lines**: Connect highs or lows to identify trend direction
            - **Triangles**: Ascending, descending, or symmetrical consolidation patterns
            - **Head & Shoulders**: Reversal pattern indicating trend change
            - **Double Top/Bottom**: Reversal patterns at key resistance/support levels
            """)
    
    with tab3:
        st.subheader("Investment Strategies")
        
        with st.expander("Risk Management"):
            st.markdown("""
            - **Position Sizing**: Never risk more than 1-5% of portfolio on a single trade
            - **Stop Losses**: Predefined exit points to limit losses
            - **Take Profits**: Secure gains at predetermined levels
            - **Diversification**: Spread risk across multiple assets
            - **Dollar-Cost Averaging**: Regular purchases regardless of price
            """)
        
        with st.expander("Portfolio Allocation Strategies"):
            st.markdown("""
            **Conservative (Low Risk)**:
            - 60-70% BTC/ETH
            - 20-30% Top 10 altcoins
            - 10-20% Stablecoins
            
            **Balanced (Medium Risk)**:
            - 40-50% BTC/ETH
            - 30-40% Top 20 altcoins
            - 10-20% Small/mid caps
            - 5-10% Stablecoins
            
            **Aggressive (High Risk)**:
            - 20-30% BTC/ETH
            - 30-40% Altcoins
            - 30-40% Small caps/DeFi
            - 5-10% Stablecoins
            """)

elif st.session_state.page == "portfolio":
    ### PORTFOLIO ANALYSIS PAGE ###
    st.header("📊 Portfolio Analysis")
    
    if 'portfolio_summary' not in st.session_state:
        st.info("🔍 Please analyze your portfolio first using the sidebar button to see detailed insights.")
        
        # Show basic portfolio info if available from Binance
        if st.session_state.get('binance_api_key') and st.session_state.get('binance_api_secret'):
            try:
                client = Client(st.session_state.binance_api_key, st.session_state.binance_api_secret)
                account_info = client.get_account()
                balances = [b for b in account_info['balances'] if float(b['free']) > 0.00001]
                
                st.subheader("📋 Current Holdings")
                holdings_data = []
                for balance in balances[:10]:  # Show top 10
                    coin = balance['asset']
                    amount = float(balance['free'])
                    holdings_data.append({"Asset": coin, "Amount": f"{amount:,.6f}"})
                
                if holdings_data:
                    st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
                else:
                    st.warning("No significant holdings found.")
                    
            except Exception as e:
                st.error(f"Could not fetch basic portfolio data: {e}")
    else:
        # Show full portfolio analysis
        portfolio_df = st.session_state.portfolio_summary['dataframe']
        
        # Portfolio Health Score
        health_data = calculate_portfolio_health(portfolio_df)
        if health_data:
            st.subheader("🏥 Portfolio Health Score")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                score_color = "green" if health_data['overall_score'] >= 70 else "orange" if health_data['overall_score'] >= 50 else "red"
                st.metric("Overall Health", f"{health_data['overall_score']}/100")
            with col2:
                st.metric("Diversification", f"{health_data['diversification_score']}/100")
            with col3:
                st.metric("Performance", f"{health_data['performance_score']}/100")
            with col4:
                st.metric("Risk Management", f"{health_data['risk_score']}/100")
            with col5:
                st.metric("News Sentiment", f"{health_data['sentiment_score']}/100")
            
            # Health recommendations
            st.subheader("💡 Health Recommendations")
            for rec in health_data['recommendations']:
                st.info(f"• {rec}")
        
        # Portfolio visualization
        st.subheader("📈 Portfolio Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            # Allocation pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=portfolio_df['Asset'], 
                values=portfolio_df['USD Value'], 
                hole=.3, 
                textinfo='percent+label'
            )])
            fig_pie.update_layout(title_text='Portfolio Allocation by USD Value')
            st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_pie_1")
        
        with col2:
            # Performance bar chart
            fig_bar = px.bar(
                portfolio_df, 
                x='Asset', 
                y='24h Perf (%)', 
                color='24h Perf (%)',
                color_continuous_scale='RdYlGn',
                title='24h Performance by Asset'
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="portfolio_bar_1")
        
        # Detailed portfolio table
        st.subheader("📋 Detailed Holdings")
        display_df = portfolio_df.copy()
        display_df['USD Value'] = display_df['USD Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
        
        cols_to_show = ['Asset', 'USD Value', '24h Perf (%)', '7d Perf (%)', 'RSI', 'News Sentiment', 'Market Cap']
        st.dataframe(display_df[cols_to_show], use_container_width=True)

# --- MAIN APP LOGIC FLOW (Updated) ---
if analyze_button and 'stage' not in st.session_state:
    # API keys are already loaded into session state in sidebar
    if not all([st.session_state.get('openai_api_key'), st.session_state.get('binance_api_key'), st.session_state.get('binance_api_secret')]):
        st.warning("⚠️ Please provide OpenAI and Binance API keys.")
    elif not st.session_state.get('tavily_api_key'):
        st.warning("⚠️ Please provide a Tavily API key for news analysis. Continuing without it.")
    else:
        st.session_state.stage = 'data_fetching'
        
        with st.spinner("🔄 Step 1/3: Performing deep analysis on your portfolio..."):
            portfolio_summary, available_funds = get_full_portfolio_analysis(
                st.session_state.binance_api_key, 
                st.session_state.binance_api_secret, 
                st.session_state.tavily_api_key
            )
            if portfolio_summary:
                st.session_state.portfolio_summary = portfolio_summary
                st.session_state.available_funds = available_funds
                st.session_state.stage = 'analysis_running'
                st.rerun()
            else:
                st.error("❌ Could not fetch and analyze portfolio data. Please check keys/permissions and try again.")
                del st.session_state.stage

if st.session_state.get('stage') == 'analysis_running':
    with st.spinner("🤖 Step 2/3: The AI committee is deliberating based on your risk profile..."):
        ### Enhanced Strategist Prompt ###
        strategist_prompt = f"""
        You are a Chief Investment Strategist. Your task is to create a clear, actionable investment and rebalancing plan tailored to the user's risk profile.
        
        **User's Risk Profile:** {st.session_state.risk_profile}
        **Available Stablecoin Funds for Investment:** ${st.session_state.available_funds:,.2f}
        
        **--- DEEP PORTFOLIO & NEWS ANALYSIS (INPUT DATA) ---**
        {st.session_state.portfolio_summary['dataframe'].to_json(orient='records', indent=2)}
        
        **--- YOUR INSTRUCTIONS ---**
        1.  **Synthesize All Data:** Analyze the portfolio, which includes fundamental (Market Cap), technical (RSI, Moving Averages), performance, and **qualitative news sentiment** metrics.
        2.  **Adhere to Risk Profile:**
            - **Conservative:** Prioritize capital preservation. Focus on large-caps (BTC, ETH). Sell high-risk, negative-sentiment assets. Be cautious with new buys.
            - **Balanced:** Seek a mix of growth and stability. Include promising mid-caps with positive news. Rebalancing is key.
            - **Aggressive:** Aim for maximum growth. Suggest smaller, high-volatility altcoins with strong narratives and positive news. Take more risk for higher rewards.
        3.  **Provide Rich Reasoning:** In your `reasoning` for each trade, you **MUST** justify your decision by referencing specific data points from the input (e.g., "Sell due to high RSI of 75 and negative news sentiment" or "Buy based on bullish SMA crossover and positive catalyst news").
        4.  **Create Actionable Trades:** For **BUY** actions, you **MUST** suggest an `entry_price`, a `take_profit_price`, and a `stop_loss_price`.
        5.  **Fill out ALL fields** in the `FinalInvestmentPlan` model with precise, data-driven recommendations.
        """
        final_plan = get_llm_response(strategist_prompt, st.session_state.openai_api_key, FinalInvestmentPlan)
        if final_plan:
            st.session_state.final_plan = final_plan
            st.session_state.stage = 'complete'
            st.rerun()
        else:
            st.error("❌ The AI analysis failed. Please try again.")
            del st.session_state.stage # Allow re-running

if st.session_state.get('stage') == 'complete':
    st.success(f"✅ Analysis Complete! Data fetched at: {st.session_state.portfolio_summary['fetched_at']}")
    plan = st.session_state.final_plan
    
    # Switch to portfolio page to show results
    st.session_state.page = "portfolio"
    
    st.header("🎯 AI Investment Strategy")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Action Plan", "📊 Portfolio Analysis", "📄 Report"])

    with tab1:
        if plan:
            st.subheader(f"Your {st.session_state.risk_profile} Investment Plan")
            
            with st.container(border=True):
                st.markdown("### 📜 Strategy Summary")
                st.write(plan.strategy_summary)
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence Score", f"{plan.confidence_score}/10")
                col2.metric("Timeline", plan.investment_timeline)
                col3.metric("Projected Impact", plan.projected_portfolio_impact)

            st.subheader("💼 Trade Recommendations")
            st.markdown("Click `Execute on Binance ↗️` to open a pre-filled order screen for safe execution.")
            
            for trade in plan.trade_recommendations:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        action_icon = "🟢" if trade.action == "BUY" else "🔴" if trade.action == "SELL" else "🟡"
                        st.markdown(f"### {action_icon} {trade.action} {trade.coin}")
                        if trade.amount_usd: 
                            st.markdown(f"**Amount:** `${trade.amount_usd:,.2f}`")
                        
                        # Display news sentiment next to reasoning
                        news_sentiment = st.session_state.portfolio_summary['dataframe'].loc[
                            st.session_state.portfolio_summary['dataframe']['Asset'] == trade.coin, 'News Sentiment'
                        ].values
                        if len(news_sentiment) > 0 and news_sentiment[0] != "N/A":
                             sentiment_color = "green" if news_sentiment[0] == "Positive" else "red" if news_sentiment[0] == "Negative" else "gray"
                             st.markdown(f"**News Sentiment:** :{sentiment_color}[{news_sentiment[0]}]")
                        
                        st.markdown(f"**Reasoning:** *{trade.reasoning}*")

                    with col2:
                        st.write("") 
                        st.write("") 
                        asset_data = st.session_state.portfolio_summary['dataframe']
                        price_row = asset_data.loc[asset_data['Asset'] == trade.coin]
                        current_price = price_row['Price'].iloc[0] if not price_row.empty else trade.entry_price
                        trade_link = create_binance_trade_link(f"{trade.coin}USDT", trade.action, trade.amount_usd, current_price)
                        if trade_link: 
                            st.link_button("Execute on Binance ↗️", trade_link, use_container_width=True)

                    if trade.action == "BUY":
                        tp_col, sl_col, en_col = st.columns(3)
                        en_col.info(f"Entry: ${trade.entry_price or 'N/A'}")
                        tp_col.success(f"Take Profit: ${trade.take_profit_price or 'N/A'}")
                        sl_col.error(f"Stop Loss: ${trade.stop_loss_price or 'N/A'}")
                    
                    if trade.action == "SELL": 
                        st.warning("⚠️ **Note:** Selling may create a taxable event.", icon="💰")

    with tab2:
        # Portfolio health and visualization
        portfolio_df = st.session_state.portfolio_summary['dataframe']
        
        # Portfolio Health Score
        health_data = calculate_portfolio_health(portfolio_df)
        if health_data:
            st.subheader("🏥 Portfolio Health Assessment")
            
            # Create a gauge chart for overall health
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_data['overall_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Health Score"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True, key="portfolio_health_gauge")
            
            # Component scores
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Diversification", f"{health_data['diversification_score']}/100")
            with col2:
                st.metric("Performance", f"{health_data['performance_score']}/100")
            with col3:
                st.metric("Risk Management", f"{health_data['risk_score']}/100")
            with col4:
                st.metric("News Sentiment", f"{health_data['sentiment_score']}/100")
            
            # Recommendations
            st.subheader("💡 Improvement Recommendations")
            for i, rec in enumerate(health_data['recommendations'], 1):
                st.info(f"{i}. {rec}")

        # Enhanced portfolio visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio allocation pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=portfolio_df['Asset'], 
                values=portfolio_df['USD Value'], 
                hole=.3, 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
            )])
            fig_pie.update_layout(title_text='Portfolio Allocation by USD Value')
            st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_pie_2")
        
        with col2:
            # Performance scatter plot
            fig_scatter = px.scatter(
                portfolio_df, 
                x='24h Perf (%)', 
                y='7d Perf (%)',
                size='USD Value',
                color='News Sentiment',
                hover_name='Asset',
                title='Performance vs News Sentiment',
                color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue', 'N/A': 'gray'}
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scatter, use_container_width=True, key="portfolio_scatter_1")

        st.subheader("📋 Detailed Asset Analysis")
        df_display = portfolio_df.copy()
        df_display = df_display.drop(columns=['Balance', 'Price', 'Name'])
        df_display['Market Cap'] = df_display['Market Cap'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
        df_display['USD Value'] = df_display['USD Value'].apply(lambda x: f"${x:,.2f}")
        df_display['Sparkline HTML'] = df_display['Sparkline'].apply(lambda x: f'<img src="data:image/png;base64,{x}">' if x else '')
        
        # Reorder for better display
        cols_to_display = ['Asset', 'USD Value', '24h Perf (%)', '7d Perf (%)', 'Sparkline HTML', 'RSI', 'Price/SMA50', 'SMA50/SMA200', 'Market Cap', 'News Sentiment', 'News Summary']
        
        st.markdown(df_display[cols_to_display].to_html(
            escape=False, 
            formatters={
                '24h Perf (%)': '{:,.2f}%'.format, 
                '7d Perf (%)': '{:,.2f}%'.format, 
                'RSI': '{:,.1f}'.format, 
                'Price/SMA50': '{:,.2f}'.format, 
                'SMA50/SMA200': '{:,.2f}'.format,
            }
        ), unsafe_allow_html=True)

    with tab3:
        st.header("📄 Download Report")
        st.write("You can download the full analysis report as a PDF or view the raw JSON output from the AI strategist.")
        
        pdf_data = generate_pdf(plan, st.session_state.portfolio_summary, st.session_state.risk_profile)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name=f"AI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="download_pdf_report_1"
        )
        
        with st.expander("Show Raw AI Output (JSON)"):
            st.json(plan.model_dump_json(indent=2))

# --- FOOTER ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>📈 AI Investment Committee Pro | Built with ❤️ using Streamlit</p>
        <p><small>⚠️ This tool is for educational purposes only. Not financial advice. Always DYOR!</small></p>
    </div>
    """, unsafe_allow_html=True)
