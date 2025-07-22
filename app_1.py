import streamlit as st
from openai import OpenAI
import instructor  # <-- IMPORT THE INSTRUCTOR LIBRARY
import json
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from tavily import TavilyClient
import time
import plotly.graph_objects as go

load_dotenv()

# --- Pydantic Models for Structured Responses ---
# These models ensure the AI provides consistent, structured output.

class SearchQueriesModel(BaseModel):
    """The model for generating web search queries."""
    queries: List[str] = Field(..., description="A list of 5-7 web search queries to gather intelligence.")

class MarketAnalysisReport(BaseModel):
    overall_market_sentiment: str = Field(..., description="Overall market sentiment (e.g., Bullish, Bearish, Neutral, Fearful).")
    market_trend: str = Field(..., description="Current market trend (e.g., Uptrend, Downtrend, Sideways Consolidation).")
    top_performing_coins: List[str] = Field(..., description="List of coins showing strong positive momentum recently.")
    market_volatility_assessment: str = Field(..., description="Assessment of current market volatility (e.g., High, Medium, Low).")
    key_market_insights: List[str] = Field(..., description="Bulleted list of critical insights driving the market right now.")
    recommended_investment_timing: str = Field(..., description="Suggested timing for new investments (e.g., 'Ideal for entry', 'Wait for a dip', 'High-risk entry').")

class PortfolioAnalysisReport(BaseModel):
    current_portfolio_value_usd: float = Field(..., description="The total estimated value of the portfolio in USD.")
    portfolio_24h_performance_pct: float = Field(..., description="The weighted average performance of the portfolio over the last 24 hours in percent.")
    diversification_score: int = Field(..., description="A score from 1 (poor) to 10 (excellent) assessing portfolio diversification.", ge=1, le=10)
    overweight_assets: List[str] = Field(..., description="Assets that represent an overly large portion of the portfolio, increasing risk.")
    underweight_opportunities: List[str] = Field(..., description="Asset classes or specific coins where the portfolio has low or no exposure, but which may be beneficial.")
    portfolio_risk_level: str = Field(..., description="The overall risk level of the current portfolio (e.g., Low, Medium, High, Very High).")
    rebalancing_suggestions: List[str] = Field(..., description="General suggestions for rebalancing the portfolio to improve its health.")

class RiskAssessmentReport(BaseModel):
    overall_risk_level: str = Field(..., description="The overall risk assessment for making new investments now (LOW, MEDIUM, HIGH).")
    risk_factors: List[str] = Field(..., description="Specific factors contributing to the current market risk (e.g., 'Regulatory uncertainty in the US', 'High inflation data').")
    potential_losses: str = Field(..., description="A realistic description of potential downside or losses if the market turns.")
    risk_mitigation_strategies: List[str] = Field(..., description="Actionable strategies to reduce investment risk (e.g., 'Use stop-loss orders', 'Diversify into stablecoins').")
    investment_warning_flags: List[str] = Field(..., description="Specific red flags or warnings the user should be aware of before investing.")
    recommended_position_size: str = Field(..., description="A recommendation on how much capital to risk on new trades (e.g., 'Small (1-2% of portfolio)', 'Standard (3-5%)', 'Aggressive (5-10%)').")

class ActionableTrade(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(..., description="The recommended action: BUY, SELL, or HOLD.")
    coin: str = Field(..., description="The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH').")
    amount_usd: Optional[float] = Field(None, description="The suggested amount in USD for the transaction. Not applicable for HOLD.")
    reasoning: str = Field(..., description="A concise, clear reason for the recommended action.")

class FinalInvestmentPlan(BaseModel):
    strategy_summary: str = Field(..., description="A brief summary of the overall investment strategy being recommended.")
    confidence_score: int = Field(..., description="A confidence score from 1 (low) to 10 (high) for this overall plan.", ge=1, le=10)
    trade_recommendations: List[ActionableTrade] = Field(..., description="A list of specific, actionable trades to execute.")
    projected_portfolio_impact: str = Field(..., description="How this plan, if executed, is expected to impact the portfolio's risk and potential return.")
    investment_timeline: str = Field(..., description="The suggested timeline for this investment plan (e.g., 'Short-term (1-4 weeks)', 'Medium-term (1-6 months)').")


# --- Configuration and API Setup ---
st.set_page_config(page_title="AI Investment Committee", layout="wide")

st.title("📈 AI Investment Committee for Binance")
st.markdown("### A Multi-Agent System for Smarter Portfolio Strategy")
st.warning(
    "**Disclaimer:** This is an advanced educational tool and not financial advice. "
    "All actions, especially **selling**, have tax implications and market risks. "
    "Always use read-only API keys. Cryptocurrency markets are extremely volatile."
)

# --- Helper Functions ---

def get_llm_response(prompt, api_key, response_model):
    """Generic function to call the OpenAI API with structured data extraction."""
    try:
        # THE FIX IS HERE: Patch the OpenAI client with `instructor`
        openai_client = instructor.patch(OpenAI(api_key=api_key))
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a world-class financial analyst. Provide structured, data-driven analysis based on the user's request. Your response MUST conform to the provided Pydantic model schema."},
                {"role": "user", "content": prompt}
            ],
            response_model=response_model # This now works because of instructor
        )
        return response
    except Exception as e:
        st.error(f"An error occurred with the LLM API: {e}")
        return None

def get_24h_price_change(client, symbol):
    """Fetches the 24-hour price change for a given symbol."""
    try:
        ticker = client.get_ticker(symbol=symbol)
        return float(ticker.get('priceChangePercent', 0.0))
    except Exception:
        return 0.0


def get_binance_data(api_key, api_secret):
    """Fetches comprehensive portfolio and market data from Binance."""
    try:
        client = Client(api_key, api_secret)
        account_info = client.get_account()
        
        balances = [b for b in account_info['balances'] if float(b['free']) > 0.00001]
        usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        
        portfolio_df_data = []
        total_portfolio_value = 0
        
        if usdt_balance > 1.0:
            portfolio_df_data.append({"Asset": "USDT", "Balance": usdt_balance, "USD Value": usdt_balance, "24h Perf (%)": 0.0})
            total_portfolio_value += usdt_balance
            
        with st.spinner("Calculating portfolio value and 24h performance..."):
            for asset in balances:
                coin = asset['asset']
                balance = float(asset['free'])
                
                if coin == 'USDT': continue

                if coin in ['USDC', 'BUSD', 'TUSD', 'FDUSD']:
                    usd_value = balance
                    perf_24h = 0.0
                else:
                    symbol = f"{coin}USDT"
                    try:
                        ticker = client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        usd_value = balance * price
                        perf_24h = get_24h_price_change(client, symbol)
                    except Exception:
                        continue
                
                if usd_value > 1.0: 
                    total_portfolio_value += usd_value
                    portfolio_df_data.append({"Asset": coin, "Balance": balance, "USD Value": usd_value, "24h Perf (%)": perf_24h})

        if not portfolio_df_data:
            st.warning("No assets worth over $1 found in your account.")
            return None, None, None, None
            
        portfolio_df = pd.DataFrame(portfolio_df_data)
        
        if total_portfolio_value > 0:
            portfolio_df['Weight'] = portfolio_df['USD Value'] / total_portfolio_value
            weighted_perf = (portfolio_df['Weight'] * portfolio_df['24h Perf (%)']).sum()
        else:
            weighted_perf = 0

        top_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
        market_data = [client.get_ticker(symbol=s) for s in top_coins]
        
        portfolio_summary = {
            "total_value_usd": total_portfolio_value,
            "performance_24h_pct": weighted_perf,
            "dataframe": portfolio_df
        }

        return portfolio_summary, usdt_balance, market_data, balances
    except Exception as e:
        st.error(f"An error occurred with the Binance API: {e}")
        st.info("Please ensure your API keys are correct, have 'Enable Reading' permissions, and are not restricted by IP.")
        return None, None, None, None


def generate_search_queries(market_data, portfolio_summary, available_funds, openai_api_key):
    """Use LLM to generate high-quality web search queries."""
    prompt = f"""
    You are a financial research assistant. Based on the user's crypto portfolio and market data, generate 5 highly relevant web search queries to gather intelligence for an investment decision.
    Focus on:
    - Current market-moving news (macro and crypto-specific).
    - Regulatory changes or major announcements.
    - Analysis of sentiment for the user's top holdings.
    - Emerging narratives or trends.
    - It's {datetime.now().strftime('%Y-%m-%d')}.
    
    **User's Portfolio Summary:**
    {portfolio_summary['dataframe'].to_json(orient='records')}
    **Available Funds for Investment:** ${available_funds:.2f}
    **Top Market Tickers:**
    {json.dumps(market_data, indent=2)}
    
    Return your response as a JSON object conforming to the SearchQueriesModel schema.
    """
    try:
        result = get_llm_response(prompt, openai_api_key, SearchQueriesModel)
        return result.queries if result else []
    except Exception as e:
        st.error(f"Failed to generate search queries: {e}")
        return []

def create_portfolio_donut_chart(df):
    """Creates a Plotly donut chart from the portfolio dataframe."""
    fig = go.Figure(data=[go.Pie(
        labels=df['Asset'],
        values=df['USD Value'],
        hole=.6,
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Allocation: %{percent}<extra></extra>'
    )])
    fig.update_layout(
        title_text='Portfolio Allocation',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        annotations=[dict(text='Assets', x=0.5, y=0.5, font_size=20, showarrow=False)],
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# --- Main Application Logic ---

if 'stage' not in st.session_state:
    st.session_state.stage = 'initial'

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.stage = 'initial'

with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Get yours from platform.openai.com")
    binance_api_key = st.text_input("Binance API Key (Read-Only)", type="password")
    binance_api_secret = st.text_input("Binance API Secret (Read-Only)", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="Get yours from tavily.com")

    if not all([openai_api_key, binance_api_key, binance_api_secret, tavily_api_key]):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_api_secret = os.getenv("BINANCE_API_SECRET")
        tavily_api_key = os.getenv("TAVILY_API_KEY")

    analyze_button = st.button("🚀 Analyze & Generate Strategy", use_container_width=True)
    if st.session_state.stage != 'initial':
        if st.button("Start New Analysis", use_container_width=True):
            reset_app()
            st.rerun()

if analyze_button and st.session_state.stage == 'initial':
    if not all([openai_api_key, binance_api_key, binance_api_secret, tavily_api_key]):
        st.warning("Please provide all API keys in the sidebar or set them in a .env file.")
    else:
        st.session_state.openai_api_key = openai_api_key
        st.session_state.tavily_api_key = tavily_api_key
        
        with st.spinner("Step 1/6: Fetching and analyzing your Binance portfolio..."):
            portfolio_summary, available_funds, market_data, balances = get_binance_data(binance_api_key, binance_api_secret)
            if portfolio_summary:
                st.session_state.portfolio_summary = portfolio_summary
                st.session_state.available_funds = available_funds
                st.session_state.market_data = market_data
                st.session_state.balances = balances 
                st.session_state.stage = 'data_fetched'
                st.rerun()
            else:
                st.error("Could not fetch portfolio data. Please check keys and ensure you have assets.")
                reset_app()

if st.session_state.stage in ['data_fetched', 'queries_generated', 'search_complete', 'analysis_complete', 'search_running', 'analysis_running']:
    st.header("Your Portfolio Dashboard")
    summary = st.session_state.portfolio_summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"${summary['total_value_usd']:,.2f}")
    col2.metric("24h Portfolio Change", f"{summary['performance_24h_pct']:.2f}%")
    col3.metric("Available to Invest (USDT)", f"${st.session_state.available_funds:,.2f}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(create_portfolio_donut_chart(summary['dataframe']), use_container_width=True)
    with col2:
        st.dataframe(summary['dataframe'].style.format({
            "Balance": "{:.4f}",
            "USD Value": "${:,.2f}",
            "24h Perf (%)": "{:.2f}%"
        }).apply(lambda x: ['background-color: #E2FFE2' if x['24h Perf (%)'] > 0 else 'background-color: #FFE2E2' for i in x], axis=1),
        use_container_width=True)

if st.session_state.stage == 'data_fetched':
    with st.spinner("Step 2/6: Generating web search queries..."):
        queries = generate_search_queries(
            st.session_state.market_data,
            st.session_state.portfolio_summary,
            st.session_state.available_funds,
            st.session_state.openai_api_key
        )
        if queries:
            st.session_state.search_queries = queries
            st.session_state.stage = 'queries_generated'
            st.rerun()
        else:
            st.error("Failed to generate search queries.")
            reset_app()

if st.session_state.stage == 'queries_generated':
    st.subheader("Proposed Web Search Queries")
    st.info("The AI will use these queries to gather the latest market intelligence.")
    for q in st.session_state.search_queries:
        st.markdown(f"- *{q}*")
    if st.button("Run Web Search & Analysis", use_container_width=True):
        st.session_state.stage = 'search_running'
        st.rerun()

if st.session_state.stage == 'search_running':
    with st.spinner("Step 3/6: Searching the web for the latest market intelligence..."):
        tavily_client = TavilyClient(api_key=st.session_state.tavily_api_key)
        search_context = []
        progress_bar = st.progress(0, "Starting web search...")
        
        for i, query in enumerate(st.session_state.search_queries):
            try:
                response = tavily_client.search(query, search_depth="advanced")
                search_context.append({"query": query, "results": response['results']})
                progress_bar.progress((i + 1) / len(st.session_state.search_queries), f"Searched: {query}")
                time.sleep(1) 
            except Exception as e:
                st.warning(f"Tavily search failed for '{query}': {e}")
        
        st.session_state.search_context = search_context
        st.session_state.stage = 'analysis_running'
        st.rerun()

if st.session_state.stage == 'analysis_running':
    context_data = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "portfolio_summary": st.session_state.portfolio_summary,
        "raw_balances": st.session_state.balances,
        "available_funds_usd": st.session_state.available_funds,
        "market_data": st.session_state.market_data,
        "web_search_context": st.session_state.search_context
    }

    with st.spinner("Step 4/6: Market Analyst is assessing the climate..."):
        market_prompt = f"Analyze the current crypto market based on the data below.\n{json.dumps(context_data, indent=2, default=str)}"
        st.session_state.market_report = get_llm_response(market_prompt, st.session_state.openai_api_key, MarketAnalysisReport)

    with st.spinner("Step 5/6: Portfolio Analyst is reviewing your assets..."):
        portfolio_prompt = f"Analyze the user's portfolio based on the data below.\n{json.dumps(context_data, indent=2, default=str)}"
        st.session_state.portfolio_report = get_llm_response(portfolio_prompt, st.session_state.openai_api_key, PortfolioAnalysisReport)
        
    with st.spinner("Step 6/6: Risk Assessor is evaluating the downside..."):
        risk_prompt = f"""Perform a risk assessment for a new investment, using the provided market and portfolio analysis.
        Market Report: {st.session_state.market_report.model_dump_json(indent=2) if st.session_state.market_report else "N/A"}
        Portfolio Report: {st.session_state.portfolio_report.model_dump_json(indent=2) if st.session_state.portfolio_report else "N/A"}
        Web Search Context: {json.dumps(context_data['web_search_context'], indent=2, default=str)}"""
        st.session_state.risk_report = get_llm_response(risk_prompt, st.session_state.openai_api_key, RiskAssessmentReport)

    with st.spinner("Putting it all together: Chief Strategist is creating your action plan..."):
        strategist_prompt = f"""You are a Chief Investment Strategist. Your task is to create a clear, actionable investment and rebalancing plan.
        **Your Goal:** Optimize the user's portfolio for better risk-adjusted returns.
        **Available Funds for New Investments:** ${context_data['available_funds_usd']:.2f}
        **--- CONSOLIDATED ANALYSIS ---**
        **Portfolio Data:**
        - Current Holdings: {context_data['portfolio_summary']['dataframe'].to_json(orient='records')}
        - Portfolio Analyst's Key Findings: {st.session_state.portfolio_report.model_dump_json(indent=2) if st.session_state.portfolio_report else 'N/A'}
        **Market & Risk Data:**
        - Market Analyst's Report: {st.session_state.market_report.model_dump_json(indent=2) if st.session_state.market_report else 'N/A'}
        - Risk Assessor's Report: {st.session_state.risk_report.model_dump_json(indent=2) if st.session_state.risk_report else 'N/A'}
        **Live Intelligence:**
        - Web Search Context: {json.dumps(context_data['web_search_context'], indent=2, default=str)}
        **Your Instructions:**
        1. Synthesize all the above information.
        2. Develop a coherent strategy.
        3. Generate a list of specific, actionable trades (`trade_recommendations`).
        4. For **BUY** actions, use the available USDT funds. Specify the coin and USD amount.
        5. For **SELL** actions, recommend selling a portion or all of a current holding. Specify the coin and USD amount to sell.
        6. For **HOLD** actions, justify why the asset should be kept.
        7. Provide a concise reasoning for EACH action.
        8. Fill out all fields in the `FinalInvestmentPlan` model.
        """
        st.session_state.final_plan = get_llm_response(strategist_prompt, st.session_state.openai_api_key, FinalInvestmentPlan)
        st.session_state.stage = 'analysis_complete'
        st.rerun()

if st.session_state.stage == 'analysis_complete':
    st.balloons()
    st.success("Analysis Complete! Here is your personalized investment strategy.")
    plan = st.session_state.final_plan

    if plan:
        st.header("🎯 Your Actionable Investment Plan")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy Confidence", f"{plan.confidence_score}/10")
        col2.metric("Investment Timeline", plan.investment_timeline)
        
        with st.container(border=True):
            st.subheader("📜 Strategy Summary")
            st.write(plan.strategy_summary)
            st.subheader("Projected Impact")
            st.info(plan.projected_portfolio_impact)

        st.subheader("Trade Recommendations")

        actions_config = {"BUY": {"icon": "🟢", "color": "#28a745"}, "SELL": {"icon": "🔴", "color": "#dc3545"}, "HOLD": {"icon": "🟡", "color": "#ffc107"}}

        for trade in plan.trade_recommendations:
            config = actions_config[trade.action]
            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"<h3 style='color:{config['color']};'>{config['icon']} {trade.action}</h3>", unsafe_allow_html=True)
                    st.markdown(f"**{trade.coin}**")
                    if trade.amount_usd:
                        st.markdown(f"**${trade.amount_usd:,.2f}**")
                with col2:
                    st.markdown(f"**Reasoning:**")
                    st.write(trade.reasoning)
    else:
        st.error("The final strategy could not be generated. This might be due to an API error or insufficient data.")

    with st.expander("🔍 Show Detailed Agent Reports & Web Search Data"):
        st.header("Individual Agent Analyses")
        if st.session_state.get('market_report'):
            st.subheader("Market Analyst Report")
            st.json(st.session_state.market_report.model_dump_json(indent=2))
        if st.session_state.get('portfolio_report'):
            st.subheader("Portfolio Analyst Report")
            st.json(st.session_state.portfolio_report.model_dump_json(indent=2))
        if st.session_state.get('risk_report'):
            st.subheader("Risk Assessor Report")
            st.json(st.session_state.risk_report.model_dump_json(indent=2))

        st.header("🌐 Web Search Context Used")
        if st.session_state.get('search_context'):
            for item in st.session_state.search_context:
                with st.container(border=True):
                    st.markdown(f"**Query:** *{item['query']}*")
                    for result in item['results'][:3]: 
                        st.markdown(f" - [{result['title']}]({result['url']})")