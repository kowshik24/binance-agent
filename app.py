import streamlit as st
from openai import OpenAI
import json
from binance.client import Client
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List, Optional
from tavily import TavilyClient
import time
from datetime import timedelta
load_dotenv()

# --- Pydantic Models for Structured Responses ---

class MarketAnalysisReport(BaseModel):
    overall_market_sentiment: str
    market_trend: str
    top_performing_coins: List[str]
    market_volatility_assessment: str
    key_market_insights: List[str]
    recommended_investment_timing: str

class PortfolioAnalysisReport(BaseModel):
    current_portfolio_value_usd: float
    diversification_score: int  # 1-10
    overweight_assets: List[str]
    underweight_opportunities: List[str]
    portfolio_risk_level: str
    rebalancing_suggestions: List[str]

class RiskAssessmentReport(BaseModel):
    overall_risk_level: str  # LOW, MEDIUM, HIGH
    risk_factors: List[str]
    potential_losses: str
    risk_mitigation_strategies: List[str]
    investment_warning_flags: List[str]
    recommended_position_size: str

class FinalInvestmentSuggestion(BaseModel):
    final_decision: str  # INVEST, HOLD, AVOID
    suggested_coin: Optional[str]
    investment_amount_usd: Optional[float]
    confidence_score: int  # 1-10
    reasoning_summary: str
    potential_actions: List[str]
    expected_roi_range: Optional[str]
    investment_timeline: Optional[str]

class SearchQueriesModel(BaseModel):
    queries: List[str]

# --- Configuration and API Setup ---
st.set_page_config(page_title="AI Investment Committee", layout="wide")

st.title("AI Investment Committee for Binance")
st.markdown("### Using a Multi-Agent System for Smarter Suggestions")
st.warning("**Disclaimer:** This is an educational tool. Not financial advice. Use with read-only API keys. Markets are volatile.")

# --- Helper Functions ---

def get_llm_response(prompt, api_key, response_model):
    """Generic function to call the OpenAI API with structured data extraction."""
    try:
        openai_client = OpenAI(api_key=api_key)
        response = openai_client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "You are a financial expert. Provide structured analysis based on the given data."},
                {"role": "user", "content": prompt}
            ],
            text_format=response_model
        )
        return response.output_parsed
    except Exception as e:
        st.error(f"An error occurred with the LLM API: {e}")
        return None

def get_binance_data(api_key, api_secret):
    """Fetches portfolio and market data from Binance."""
    try:
        client = Client(api_key, api_secret)
        
        # Get account balances
        account_info = client.get_account()
        balances = [b for b in account_info['balances'] if float(b['free']) > 0]
        
        # Get available USDT (or other stablecoin) for funding
        usdt_balance = client.get_asset_balance(asset='USDT')['free']
        
        # Get market data for top coins
        tickers = client.get_ticker()
        top_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
        market_data = [t for t in tickers if t['symbol'] in top_coins]
        
        return balances, float(usdt_balance), market_data
    except Exception as e:
        st.error(f"An error occurred with the Binance API: {e}")
        st.info("Please ensure your API keys are correct and have 'Enable Reading' permissions.")
        return None, None, None

def generate_search_queries(market_data, balances, available_funds, openai_api_key):
    """Use LLM to generate a list of at least 5 high-quality search queries for web search, using a Pydantic model for structured output."""
    prompt = f"""
    You are a financial research assistant. Given the following user's crypto portfolio and market data, generate a list of at least 5 highly relevant, up-to-date web search queries. 
    The queries should focus on:
    - Current market-moving news or events for the top coins in the user's portfolio
    - Regulatory or macroeconomic risks affecting crypto
    - Major upcoming events (e.g., ETF approvals, forks, exchange issues)
    - Portfolio-specific risks or opportunities
    - Any urgent warnings or opportunities for the user's holdings
    - It's {datetime.now().strftime('%Y-%m-%d')}, so consider the latest trends and news.
    
    **User's Portfolio:**
    {json.dumps(balances, indent=2)}
    **Available Funds:** ${available_funds:.2f}
    **Market Data:**
    {json.dumps(market_data, indent=2)}
    
    Please return your response as a JSON object with a single key 'queries', whose value is a list of search query strings. Example: {{"queries": ["query1", "query2", ...]}}
    """
    try:
        result = get_llm_response(prompt, openai_api_key, SearchQueriesModel)
        if result and result.queries:
            return result.queries[:10]
        else:
            st.error("No queries returned from LLM.")
            return []
    except Exception as e:
        st.error(f"Failed to generate search queries: {e}")
        return []

# --- Main Application Logic ---

# Initialize session state for app flow control
if 'stage' not in st.session_state:
    st.session_state.stage = 'initial'  # Stages: initial, queries_generated, search_complete, analysis_complete
if 'search_queries' not in st.session_state:
    st.session_state.search_queries = []
if 'search_context' not in st.session_state:
    st.session_state.search_context = []
if 'balances' not in st.session_state:
    st.session_state.balances = None
if 'available_funds' not in st.session_state:
    st.session_state.available_funds = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None

# Sidebar for API keys and user inputs
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Get yours from platform.openai.com")
    binance_api_key = st.text_input("Binance API Key (Read-Only)", type="password")
    binance_api_secret = st.text_input("Binance API Secret (Read-Only)", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="Get yours from tavily.com")
    analyze_button = st.button("Analyze & Get Suggestions", use_container_width=True)

    # Load all the keys from the environment variables
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not binance_api_key:
        binance_api_key = os.getenv("BINANCE_API_KEY")
    if not binance_api_secret:
        binance_api_secret = os.getenv("BINANCE_API_SECRET")
    if not tavily_api_key:
        tavily_api_key = os.getenv("TAVILY_API_KEY")

def reset_app():
    """Reset the app to initial state"""
    st.session_state.stage = 'initial'
    st.session_state.search_queries = []
    st.session_state.search_context = []
    st.session_state.balances = None
    st.session_state.available_funds = None
    st.session_state.market_data = None

# Initial data fetch
if analyze_button:
    if not (openai_api_key and binance_api_key and binance_api_secret):
        st.warning("Please enter all API keys in the sidebar.")
    else:
        reset_app()  # Reset on new analysis
        with st.spinner("Step 1/6: Fetching your data from Binance..."):
            balances, available_funds, market_data = get_binance_data(binance_api_key, binance_api_secret)
            if balances is not None:
                st.session_state.balances = balances
                st.session_state.available_funds = available_funds
                st.session_state.market_data = market_data
                st.session_state.stage = 'queries_generated'

# Query generation and display
if st.session_state.stage == 'queries_generated':
    # --- WEB SEARCH QUERY GENERATION ---
    with st.spinner("Step 2/6: Generating web search queries for context..."):
        search_queries = generate_search_queries(
            st.session_state.market_data, 
            st.session_state.balances, 
            st.session_state.available_funds, 
            openai_api_key
        )
    if not search_queries or len(search_queries) < 3:
        st.error("Failed to generate enough search queries for web search context.")
        reset_app()
    else:
        st.session_state.search_queries = search_queries
        st.subheader("Proposed Web Search Queries")
        for i, q in enumerate(search_queries, 1):
            st.markdown(f"**{i}.** {q}")
        
        proceed_search = st.button("Run Web Search with These Queries", key="run_web_search")
        if proceed_search:
            st.session_state.stage = 'search_complete'
            st.rerun()
        else:
            st.info("Review the queries above. Click the button to proceed with web search and analysis.")

# Web search execution
if st.session_state.stage == 'search_complete':
    # --- WEB SEARCH EXECUTION ---
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        st.error("Tavily API key not found in environment. Please set TAVILY_API_KEY in your .env file.")
        reset_app()
    else:
        tavily_client = TavilyClient(api_key=tavily_api_key)
        search_context = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, query in enumerate(st.session_state.search_queries):
            status_text.text(f"Searching the web for: {query}")
            try:
                response = tavily_client.search(query)
                search_context.append({"query": query, "results": response})
                progress_bar.progress((idx + 1) / len(st.session_state.search_queries))
                time.sleep(1.5)  # Rate limit buffer
            except Exception as e:
                st.warning(f"Tavily search failed for '{query}': {e}")
                time.sleep(3)
        
        st.session_state.search_context = search_context
        status_text.text("Web search complete!")
        st.session_state.stage = 'analysis_complete'
        st.rerun()
# --- AGENT ORCHESTRATION ---
if st.session_state.stage == 'analysis_complete':
    # We have all the data we need for analysis
    market_report = None
    portfolio_report = None
    risk_report = None
    final_suggestion = None
    
    # 1. Market Analyst
    with st.spinner("Step 3/6: The Market Analyst is assessing the climate..."):
        market_prompt = f"""
        You are an expert crypto market analyst. Analyze the current market conditions and provide insights.
        \n**Context:**
        - Current Date: {datetime.now().strftime('%Y-%m-%d')}
        **Market Data:**
        {json.dumps(st.session_state.market_data, indent=2)}
        \n**Web Search Context:**
        {json.dumps(st.session_state.search_context, indent=2)}
        \nPlease provide a comprehensive market analysis including overall sentiment, trends, top performing coins, volatility assessment, key insights, and investment timing recommendations.
        """
        market_report = get_llm_response(market_prompt, openai_api_key, MarketAnalysisReport)

    # 2. Portfolio Analyst
    with st.spinner("Step 4/6: The Portfolio Analyst is reviewing your assets..."):
        portfolio_prompt = f"""
        You are an expert portfolio analyst. Analyze the user's current cryptocurrency portfolio.
        \n**Context:**
        - Available funds for new investment: ${st.session_state.available_funds:.2f}
        **User's Portfolio Data:**
        {json.dumps(st.session_state.balances, indent=2)}
        \n**Web Search Context:**
        {json.dumps(st.session_state.search_context, indent=2)}
        \nPlease provide a comprehensive portfolio analysis including current value, diversification score, overweight/underweight assets, risk level, and rebalancing suggestions.
        """
        portfolio_report = get_llm_response(portfolio_prompt, openai_api_key, PortfolioAnalysisReport)

    # 3. Risk Assessment Agent
    with st.spinner("Step 5/6: The Risk Assessor is playing devil's advocate..."):
        risk_prompt = f"""
        You are a cautious and skeptical risk management expert. Assess the risks of potential investment decisions.
        \n**Market Analyst Report:**
        Market Sentiment: {market_report.overall_market_sentiment if market_report else 'N/A'}
        Market Trend: {market_report.market_trend if market_report else 'N/A'}
        Top Performers: {market_report.top_performing_coins if market_report else 'N/A'}
        \n**Portfolio Analyst Report:**
        Portfolio Value: ${portfolio_report.current_portfolio_value_usd if portfolio_report else 0:.2f}
        Risk Level: {portfolio_report.portfolio_risk_level if portfolio_report else 'N/A'}
        Diversification Score: {portfolio_report.diversification_score if portfolio_report else 'N/A'}/10
        \n**Web Search Context:**
        {json.dumps(st.session_state.search_context, indent=2)}
        \nPlease provide a comprehensive risk assessment including overall risk level, risk factors, potential losses, mitigation strategies, warning flags, and recommended position size.
        """
        risk_report = get_llm_response(risk_prompt, openai_api_key, RiskAssessmentReport)
    
    # 4. Chief Strategist Agent
    with st.spinner("Step 6/6: The Chief Strategist is making the final call..."):
        strategist_prompt = f"""
        You are a Chief Investment Strategist making the final investment decision based on all available analysis.
        \n**Client's Goal:** Achieve a potential ROI of >5% on a new investment.
        **Available Funds for Investment:** ${st.session_state.available_funds:.2f}
        \n**--- ANALYST REPORTS ---**
        **Market Analysis:**
        - Sentiment: {market_report.overall_market_sentiment if market_report else 'N/A'}
        - Trend: {market_report.market_trend if market_report else 'N/A'}
        - Top Performers: {market_report.top_performing_coins if market_report else 'N/A'}
        - Investment Timing: {market_report.recommended_investment_timing if market_report else 'N/A'}
        \n**Portfolio Analysis:**
        - Current Value: ${portfolio_report.current_portfolio_value_usd if portfolio_report else 0:.2f}
        - Diversification: {portfolio_report.diversification_score if portfolio_report else 'N/A'}/10
        - Risk Level: {portfolio_report.portfolio_risk_level if portfolio_report else 'N/A'}
        \n**Risk Assessment:**
        - Overall Risk: {risk_report.overall_risk_level if risk_report else 'N/A'}
        - Risk Factors: {risk_report.risk_factors if risk_report else 'N/A'}
        - Position Size Rec: {risk_report.recommended_position_size if risk_report else 'N/A'}
        \n**Web Search Context:**
        {json.dumps(st.session_state.search_context, indent=2)}
        \nPlease provide a final investment decision with specific coin recommendation, investment amount, confidence score, reasoning, actionable steps, expected ROI range, and timeline.
        """
        final_suggestion = get_llm_response(strategist_prompt, openai_api_key, FinalInvestmentSuggestion)

    st.success("Analysis Complete! Here is the final recommendation.")

    # --- Display Final Results ---
    if final_suggestion:
        st.header("🎯 Final Investment Suggestion")
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", final_suggestion.final_decision)
        col2.metric("Confidence", f"{final_suggestion.confidence_score}/10")
        if final_suggestion.final_decision == "INVEST" and final_suggestion.suggested_coin:
            col3.metric(f"Invest in {final_suggestion.suggested_coin}", f"${final_suggestion.investment_amount_usd:.2f}")

        st.subheader("Reasoning")
        st.info(final_suggestion.reasoning_summary)

        st.subheader("Actionable Steps")
        for action in final_suggestion.potential_actions:
            st.markdown(f"- {action}")
            
        if final_suggestion.expected_roi_range:
            st.subheader("Expected ROI Range")
            st.success(final_suggestion.expected_roi_range)
            
        if final_suggestion.investment_timeline:
            st.subheader("Investment Timeline")
            st.info(final_suggestion.investment_timeline)

    # --- Display Individual Agent Reports for Transparency ---
    st.markdown("---")
    st.header("🔍 Detailed Agent Reports")
    
    # Market Analyst Report with better UI
    with st.expander("Market Analyst Report"):
        if market_report:
            # Market Sentiment section
            st.markdown("<p style='color:#555;font-size:14px;font-weight:500;margin-bottom:0'>Market Sentiment</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#333;font-size:22px;margin-top:0;margin-bottom:20px'>{market_report.overall_market_sentiment}</h2>", unsafe_allow_html=True)
            
            # Market Trend section
            st.markdown("<p style='color:#555;font-size:14px;font-weight:500;margin-bottom:0'>Market Trend</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#333;font-size:22px;margin-top:0;margin-bottom:20px'>{market_report.market_trend}</h2>", unsafe_allow_html=True)
            
            # Volatility Assessment section
            st.markdown("<p style='color:#555;font-size:14px;font-weight:500;margin-bottom:0'>Volatility Assessment</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#333;font-size:22px;margin-top:0;margin-bottom:20px'>{market_report.market_volatility_assessment}</h2>", unsafe_allow_html=True)
            
            # Recommended Timing section
            st.markdown("<p style='color:#555;font-size:14px;font-weight:500;margin-bottom:0'>Recommended Timing</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#333;font-size:22px;margin-top:0;margin-bottom:20px'>{market_report.recommended_investment_timing}</h2>", unsafe_allow_html=True)
            
            # Top Performing Coins section
            st.markdown("<h3 style='color:#333;font-size:18px;margin-top:30px'>Top Performing Coins</h3>", unsafe_allow_html=True)
            coins_html = ""
            for coin in market_report.top_performing_coins:
                coins_html += f"<span style='background-color:#E8F4F9;padding:5px 10px;margin:5px;border-radius:15px;display:inline-block;font-weight:500'>{coin}</span> "
            st.markdown(f"<div style='margin:10px 0'>{coins_html}</div>", unsafe_allow_html=True)
                
            # Key Market Insights section
            st.markdown("<h3 style='color:#333;font-size:18px;margin-top:30px'>Key Market Insights</h3>", unsafe_allow_html=True)
            for idx, insight in enumerate(market_report.key_market_insights):
                st.markdown(f"<div style='background-color:#f8f9fa;padding:10px 15px;margin:8px 0;border-radius:5px;border-left:4px solid #4e8098'>{insight}</div>", unsafe_allow_html=True)
        else:
            st.error("Market analysis failed to generate.")
    
    # Portfolio Analyst Report with better UI
    with st.expander("Portfolio Analyst Report"):
        if portfolio_report:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Value", f"${portfolio_report.current_portfolio_value_usd:.2f}")
            with col2:
                st.metric("Diversification Score", f"{portfolio_report.diversification_score}/10")
            with col3:
                st.metric("Risk Level", portfolio_report.portfolio_risk_level)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Overweight Assets")
                for asset in portfolio_report.overweight_assets:
                    st.markdown(f"<div style='background-color:#FFE2E2;padding:8px;margin:5px 0;border-radius:5px;border-left:4px solid #FF7575'>⚠️ {asset}</div>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Underweight Opportunities")
                for asset in portfolio_report.underweight_opportunities:
                    st.markdown(f"<div style='background-color:#E2FFE2;padding:8px;margin:5px 0;border-radius:5px;border-left:4px solid #75FF75'>💡 {asset}</div>", unsafe_allow_html=True)
            
            st.subheader("Rebalancing Suggestions")
            for suggestion in portfolio_report.rebalancing_suggestions:
                st.markdown(f"<div style='background-color:#f8f9fa;padding:10px;margin:5px 0;border-radius:5px;border-left:4px solid #9575FF'>🔄 {suggestion}</div>", unsafe_allow_html=True)
        else:
            st.error("Portfolio analysis failed to generate.")
    
    # Risk Assessment Report with better UI
    with st.expander("Risk Assessment Report"):
        if risk_report:
            # Create a color based on risk level
            risk_color = "#FF5252" if risk_report.overall_risk_level == "HIGH" else "#FFC107" if risk_report.overall_risk_level == "MEDIUM" else "#4CAF50"
            
            st.markdown(f"<div style='background-color:{risk_color}25;padding:15px;border-radius:10px;border:2px solid {risk_color}'><h3 style='color:{risk_color};margin:0'>Overall Risk: {risk_report.overall_risk_level}</h3></div>", unsafe_allow_html=True)
            
            st.subheader("Potential Losses")
            st.info(risk_report.potential_losses)
            
            st.subheader("Risk Factors")
            for factor in risk_report.risk_factors:
                st.markdown(f"<div style='background-color:#ffecb3;padding:8px;margin:5px 0;border-radius:5px;border-left:4px solid #FFA000'>⚠️ {factor}</div>", unsafe_allow_html=True)
            
            st.subheader("Risk Mitigation Strategies")
            for strategy in risk_report.risk_mitigation_strategies:
                st.markdown(f"<div style='background-color:#E8F5E9;padding:8px;margin:5px 0;border-radius:5px;border-left:4px solid #43A047'>🛡️ {strategy}</div>", unsafe_allow_html=True)
            
            st.subheader("Investment Warning Flags")
            for flag in risk_report.investment_warning_flags:
                st.markdown(f"<div style='background-color:#FFEBEE;padding:8px;margin:5px 0;border-radius:5px;border-left:4px solid #E53935'>🚩 {flag}</div>", unsafe_allow_html=True)
            
            st.subheader("Recommended Position Size")
            st.success(risk_report.recommended_position_size)
        else:
            st.error("Risk assessment failed to generate.")
    
    # Display Web Search Context in a more readable format
    st.markdown("---")
    st.header("🌐 Web Search Context Used")
    
    if st.session_state.search_context:
        tabs = st.tabs([f"Query {i+1}" for i in range(len(st.session_state.search_context))])
        
        for i, (tab, search_item) in enumerate(zip(tabs, st.session_state.search_context)):
            with tab:
                query = search_item.get('query', 'No query')
                results = search_item.get('results', {})
                
                st.markdown(f"<div style='background-color:#E3F2FD;padding:10px;border-radius:5px;margin-bottom:15px'><h3 style='margin:0'>🔍 {query}</h3></div>", unsafe_allow_html=True)
                
                if 'answer' in results and results['answer']:
                    st.markdown("### Summary")
                    st.markdown(f"<div style='background-color:#EDE7F6;padding:15px;border-radius:10px;margin:10px 0'>{results['answer']}</div>", unsafe_allow_html=True)
                
                if 'results' in results and results['results']:
                    st.markdown("### Sources")
                    for j, result in enumerate(results['results']):
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if 'favicon' in result and result['favicon']:
                                    st.markdown(f"<img src='{result['favicon']}' style='max-height:20px;max-width:20px'>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"**[{result.get('title', 'No title')}]({result.get('url', '#')})**")
                        
                        st.markdown(f"<div style='background-color:#F5F5F5;padding:10px;border-radius:5px;margin:5px 0 15px 0'>{result.get('content', 'No content available')}</div>", unsafe_allow_html=True)
    else:
        st.info("No web search context available.")
    
    # Reset button
    if st.button("Start New Analysis", key="reset_analysis"):
        reset_app()
        st.rerun()