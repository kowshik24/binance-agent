import streamlit as st
from openai import OpenAI
import instructor
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
import pandas_ta as ta
from pycoingecko import CoinGeckoAPI
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import base64

# --- Initial Setup ---
load_dotenv()
st.set_page_config(page_title="AI Investment Committee Pro", layout="wide")

# --- Pydantic Models for Enhanced, Structured Responses ---

class ActionableTrade(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(..., description="The recommended action.")
    coin: str = Field(..., description="The cryptocurrency ticker symbol (e.g., 'BTC').")
    amount_usd: Optional[float] = Field(None, description="Suggested USD amount for the transaction.")
    reasoning: str = Field(..., description="Clear, concise reason for the recommended action.")
    entry_price: Optional[float] = Field(None, description="Suggested entry price for a BUY order.")
    take_profit_price: Optional[float] = Field(None, description="Suggested price to take profit for a BUY order.")
    stop_loss_price: Optional[float] = Field(None, description="Suggested price to set a stop-loss for a BUY order.")

class FinalInvestmentPlan(BaseModel):
    strategy_summary: str = Field(..., description="Brief summary of the overall investment strategy.")
    confidence_score: int = Field(..., ge=1, le=10, description="Confidence score (1-10) for this plan.")
    trade_recommendations: List[ActionableTrade] = Field(..., description="List of specific, actionable trades.")
    projected_portfolio_impact: str = Field(..., description="Expected impact on the portfolio's risk and return.")
    investment_timeline: str = Field(..., description="Suggested timeline for this plan (e.g., 'Short-term (1-4 weeks)').")

# --- API Clients and Helper Functions ---
cg = CoinGeckoAPI()

@st.cache_data
def get_coin_id_map():
    """Fetches and caches the mapping from coin symbols to CoinGecko IDs."""
    try:
        coins = cg.get_coins_list()
        return {c['symbol'].upper(): c['id'] for c in coins}
    except Exception:
        return {}

COIN_ID_MAP = get_coin_id_map()

def get_llm_response(prompt, api_key, response_model):
    """Function to call OpenAI API with structured data extraction via Instructor."""
    try:
        openai_client = instructor.patch(OpenAI(api_key=api_key))
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a world-class financial analyst. Provide structured, data-driven analysis based on the user's request, conforming strictly to the provided Pydantic model schema."},
                {"role": "user", "content": prompt}
            ],
            response_model=response_model
        )
        return response
    except Exception as e:
        st.error(f"An error occurred with the LLM API: {e}")
        return None

@st.cache_data(ttl=300)
def get_advanced_binance_data(api_key, api_secret):
    """Fetches comprehensive portfolio, technical, and fundamental data."""
    try:
        client = Client(api_key, api_secret)
        account_info = client.get_account()
        balances = [b for b in account_info['balances'] if float(b['free']) > 0.00001]
        usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        
        portfolio_data = []
        total_portfolio_value = 0

        for asset in balances:
            coin = asset['asset']
            balance = float(asset['free'])
            symbol = f"{coin}USDT"
            
            try:
                price_info = client.get_symbol_ticker(symbol=symbol)
                price = float(price_info['price'])
                usd_value = balance * price

                if usd_value < 1.0: continue

                klines_4h = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, "3 months ago UTC")
                df = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
                df['close'] = pd.to_numeric(df['close'])
                
                rsi = df.ta.rsi().iloc[-1] if not df.ta.rsi().empty else None
                sma50 = df.ta.sma(50).iloc[-1] if not df.ta.sma(50).empty else None
                sma200 = df.ta.sma(200).iloc[-1] if not df.ta.sma(200).empty else None
                
                klines_7d = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "8 day ago UTC")
                df_7d = pd.DataFrame(klines_7d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
                df_7d['close'] = pd.to_numeric(df_7d['close'])
                perf_24h = (df_7d['close'].iloc[-1] / df_7d['close'].iloc[-2] - 1) * 100 if len(df_7d) > 1 else 0
                perf_7d = (df_7d['close'].iloc[-1] / df_7d['close'].iloc[0] - 1) * 100 if len(df_7d) > 0 else 0
                sparkline = create_sparkline(df_7d['close'])

                coin_id = COIN_ID_MAP.get(coin)
                market_cap = None
                if coin_id:
                    coin_data = cg.get_coin_by_id(coin_id, market_data='true')
                    market_cap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd')

                total_portfolio_value += usd_value
                portfolio_data.append({
                    "Asset": coin, "Balance": balance, "Price": price, "USD Value": usd_value, "24h Perf (%)": perf_24h, "7d Perf (%)": perf_7d,
                    "RSI": rsi, "Price/SMA50": price / sma50 if sma50 and sma50 > 0 else None, "SMA50/SMA200": sma50 / sma200 if sma50 and sma200 and sma200 > 0 else None,
                    "Market Cap": market_cap, "Sparkline": sparkline
                })

            except Exception:
                if coin.endswith('UP') or coin.endswith('DOWN') or coin in ['USDT', 'USDC', 'FDUSD']:
                    continue
        
        if usdt_balance > 1.0:
            total_portfolio_value += usdt_balance
            portfolio_data.append({"Asset": "USDT", "Balance": usdt_balance, "Price": 1.0, "USD Value": usdt_balance, "24h Perf (%)": 0, "7d Perf (%)": 0, "RSI": None, "Price/SMA50": None, "SMA50/SMA200": None, "Market Cap": None, "Sparkline": ""})
            
        if not portfolio_data: return None, None
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_summary = {"total_value_usd": total_portfolio_value, "dataframe": portfolio_df}
        
        return portfolio_summary, usdt_balance

    except Exception as e:
        st.error(f"An error occurred with the Binance API: {e}")
        return None, None

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
st.markdown("An advanced multi-agent system with deep data analysis for smarter portfolio strategy.")
st.info("💡 **Feature:** Action Plan now includes direct links to pre-fill trades on Binance for safe review and execution.")

with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="platform.openai.com")
    binance_api_key = st.text_input("Binance API Key (Read-Only)", type="password")
    binance_api_secret = st.text_input("Binance API Secret (Read-Only)", type="password")

    if not all([openai_api_key, binance_api_key, binance_api_secret]):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_api_secret = os.getenv("BINANCE_API_SECRET")

    st.session_state.risk_profile = st.selectbox(
        "Select Your Risk Profile", ["Conservative", "Balanced", "Aggressive"], index=1
    )

    analyze_button = st.button("🚀 Analyze & Generate Strategy", use_container_width=True)
    if st.session_state.get('stage', 'initial') != 'initial':
        if st.button("Start New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

if analyze_button and 'stage' not in st.session_state:
    if not all([openai_api_key, binance_api_key, binance_api_secret]):
        st.warning("Please provide all API keys in the sidebar or set them in a .env file.")
    else:
        st.session_state.openai_api_key = openai_api_key
        st.session_state.stage = 'data_fetching'
        
        with st.spinner("Step 1/3: Fetching and performing deep analysis on your portfolio..."):
            portfolio_summary, available_funds = get_advanced_binance_data(binance_api_key, binance_api_secret)
            if portfolio_summary:
                st.session_state.portfolio_summary = portfolio_summary
                st.session_state.available_funds = available_funds
                st.session_state.stage = 'analysis_running'
                st.rerun()
            else:
                st.error("Could not fetch portfolio data. Please check keys/permissions.")
                del st.session_state.stage

if st.session_state.get('stage') == 'analysis_running':
    with st.spinner("Step 2/3: The AI committee is deliberating based on your risk profile..."):
        strategist_prompt = f"""
        You are a Chief Investment Strategist. Your task is to create a clear, actionable investment and rebalancing plan tailored to the user's risk profile.
        **User's Risk Profile:** {st.session_state.risk_profile}
        **Available Funds for Investment:** ${st.session_state.available_funds:,.2f}
        **--- DEEP PORTFOLIO ANALYSIS (INPUT DATA) ---**
        {st.session_state.portfolio_summary['dataframe'].to_json(orient='records', indent=2)}
        **--- YOUR INSTRUCTIONS ---**
        1. **Synthesize All Data:** Analyze the portfolio data, which includes fundamental (Market Cap), technical (RSI, Moving Averages), and performance metrics.
        2. **Adhere to Risk Profile:**
            - **Conservative:** Prioritize capital preservation. Focus on large-caps (BTC, ETH). Sell high-risk assets. Be cautious with new buys.
            - **Balanced:** Seek a mix of growth and stability. Include promising mid-caps. Rebalancing is key.
            - **Aggressive:** Aim for maximum growth. Suggest smaller, high-volatility altcoins with strong narratives. Take more risk for higher rewards.
        3. **Generate a Coherent Strategy:** Briefly summarize your overarching plan.
        4. **Create Actionable Trades:** For **BUY** actions, you **MUST** suggest an `entry_price`, a `take_profit_price`, and a `stop_loss_price`. For **SELL** actions, recommend selling to de-risk, take profits, or reallocate. For **HOLD** actions, justify why the asset's risk/reward is appropriate.
        5. **Fill out ALL fields** in the `FinalInvestmentPlan` model with precise, data-driven recommendations.
        """
        final_plan = get_llm_response(strategist_prompt, st.session_state.openai_api_key, FinalInvestmentPlan)
        if final_plan:
            st.session_state.final_plan = final_plan
            st.session_state.stage = 'complete'
            st.rerun()
        else:
            st.error("The AI analysis failed. Please try again.")
            st.session_state.stage = 'initial'

if st.session_state.get('stage') == 'complete':
    st.success("Analysis Complete!")
    plan = st.session_state.get('final_plan')
    
    tab1, tab2, tab3 = st.tabs(["🎯 Action Plan", "📊 Portfolio Deep-Dive", "📄 Report"])

    with tab1:
        if plan:
            st.header(f"Your {st.session_state.risk_profile} Investment Plan")
            
            with st.container(border=True):
                st.subheader("📜 Strategy Summary")
                st.write(plan.strategy_summary)

            st.subheader("Trade Recommendations")
            st.markdown("Click `Execute on Binance ↗️` to open a pre-filled order screen for safe execution.")
            
            for trade in plan.trade_recommendations:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        action_icon = "🟢" if trade.action == "BUY" else "🔴" if trade.action == "SELL" else "🟡"
                        st.markdown(f"<h5>{action_icon} {trade.action} {trade.coin}</h5>", unsafe_allow_html=True)
                        if trade.amount_usd: st.markdown(f"**Amount:** `${trade.amount_usd:,.2f}`")
                        st.markdown(f"**Reasoning:** *{trade.reasoning}*")

                    with col2:
                        st.write("") 
                        st.write("") 
                        asset_data = st.session_state.portfolio_summary['dataframe']
                        price_row = asset_data.loc[asset_data['Asset'] == trade.coin]
                        current_price = price_row['Price'].iloc[0] if not price_row.empty else trade.entry_price
                        trade_link = create_binance_trade_link(f"{trade.coin}USDT", trade.action, trade.amount_usd, current_price)
                        if trade_link: st.link_button("Execute on Binance ↗️", trade_link, use_container_width=True)

                    if trade.action == "BUY":
                        tp_col, sl_col, en_col = st.columns(3)
                        en_col.info(f"Entry: ${trade.entry_price or 'N/A'}")
                        tp_col.success(f"Take Profit: ${trade.take_profit_price or 'N/A'}")
                        sl_col.error(f"Stop Loss: ${trade.stop_loss_price or 'N/A'}")
                    
                    if trade.action == "SELL": st.warning("⚠️ **Note:** Selling may create a taxable event.", icon="💰")

            st.header("🔬 What-If Scenario Analysis")
            with st.form("what_if_form"):
                what_if_query = st.text_area("Ask a follow-up question:", placeholder="e.g., 'Generate a plan that only invests in AI coins.' or 'What if I want to be more conservative?'")
                submitted = st.form_submit_button("Generate Alternative Plan")
                if submitted and what_if_query:
                    with st.spinner("The Chief Strategist is re-evaluating..."):
                        what_if_prompt = f"The user has an initial plan but wants an alternative. **Original Portfolio:** {st.session_state.portfolio_summary['dataframe'].to_json(orient='records')}, **Risk Profile:** {st.session_state.risk_profile}, **Available Funds:** ${st.session_state.available_funds:,.2f}. **USER'S NEW REQUEST:** \"{what_if_query}\". Generate a *new* `FinalInvestmentPlan` to address this."
                        st.session_state.alternative_plan = get_llm_response(what_if_prompt, st.session_state.openai_api_key, FinalInvestmentPlan)
            
            if 'alternative_plan' in st.session_state and st.session_state.alternative_plan:
                st.subheader("Alternative Plan")
                alt_plan = st.session_state.alternative_plan
                st.write(f"**Summary:** {alt_plan.strategy_summary}")
                for trade in alt_plan.trade_recommendations: st.markdown(f"- **{trade.action} {trade.coin}**: {trade.reasoning}")

            st.subheader("⚠️ A Note on Fund Transfers")
            st.info("For your security, this app cannot transfer funds (e.g., Spot to Funding). Please do this manually on Binance.")

    with tab2:
        st.header("Portfolio Deep-Dive")
        df_display = st.session_state.portfolio_summary['dataframe'].copy()
        df_display = df_display.drop(columns=['Balance', 'Price'])
        df_display['Market Cap'] = df_display['Market Cap'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
        df_display['USD Value'] = df_display['USD Value'].apply(lambda x: f"${x:,.2f}")
        df_display['Sparkline HTML'] = df_display['Sparkline'].apply(lambda x: f'<img src="data:image/png;base64,{x}">' if x else '')
        
        st.markdown(df_display[['Asset', 'USD Value', '24h Perf (%)', '7d Perf (%)', 'Sparkline HTML', 'RSI', 'Price/SMA50', 'SMA50/SMA200', 'Market Cap']].to_html(escape=False, formatters={'24h Perf (%)': '{:,.2f}%'.format, '7d Perf (%)': '{:,.2f}%'.format, 'RSI': '{:,.1f}'.format, 'Price/SMA50': '{:,.2f}'.format, 'SMA50/SMA200': '{:,.2f}'.format,}), unsafe_allow_html=True)

    with tab3:
        st.header("Download Report")
        st.write("You can download the full analysis report as a PDF or view the raw JSON output from the AI strategist.")
        
        pdf_data = generate_pdf(plan, st.session_state.portfolio_summary, st.session_state.risk_profile)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name=f"AI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
        
        with st.expander("Show Raw AI Output (JSON)"):
            st.json(plan.model_dump_json(indent=2))