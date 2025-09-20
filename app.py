import streamlit as st
import requests
import pandas as pd
import google.generativeai as genai
import json
import re
import os
import time
from pydantic import BaseModel, Field
import textwrap
import pathway as pw
from pathway import debug

# --- 1. Setup & Configuration ---
st.set_page_config(
    page_title="TaxWise AI",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Sidebar for API Key Configuration ---
st.sidebar.title("Configuration")
st.sidebar.markdown(
    """
1.  Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Enter the key below to activate the app.
"""
)
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key", key="gemini_api_key", type="password"
)
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
Optionally, provide an [ExchangeRate-API](https://www.exchangerate-api.com/) key for higher reliability.
"""
)
exchange_rate_api_key = st.sidebar.text_input(
    "ExchangeRate-API Key",
    key="exchange_rate_api_key",
    type="password",
    help="Enter your v6 API key here."
)
st.sidebar.markdown("---")


# --- 2. Pydantic & Pathway Schemas ---
class QueryDetails(BaseModel):
    product: str = Field(
        ..., description="The product category, e.g., 'laptop', 'smartphone'"
    )
    country: str = Field(
        ..., description="The country of origin, e.g., 'USA', 'Japan'"
    )
    price_usd: float = Field(
        ..., description="The price of the product in USD"
    )
    destination: str = Field(
        "India", description="The destination country for import, e.g., 'India', 'Germany'"
    )

class TaxRule(pw.Schema):
    product_category: str
    origin_country: str
    destination_country: str
    tax_rate: float
    fixed_fee_usd: float

# --- 3. Core Functions ---
@st.cache_data(ttl=3600)  # Cache the exchange rate for 1 hour
def get_live_exchange_rate_inr(api_key=None):
    """Fetches the live USD to INR exchange rate with a reliable fallback."""
    if api_key:
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
        st.session_state['exchange_rate_source'] = 'Live API (Authenticated)'
    else:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        st.session_state['exchange_rate_source'] = 'Live API (Free Tier)'

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success" and "rates" in data and "INR" in data["rates"]:
            return data["rates"]["INR"]
        else:
            # --- IMPROVED ERROR HANDLING ---
            # Show the full API error response for better debugging instead of a generic message.
            error_details = data.get("error-type", f"Full API Response: {data}")
            st.warning(f"Exchange rate API error: '{error_details}'. Using fallback rate.")
            st.session_state['exchange_rate_source'] = 'Fallback (API Error)'
            return 83.50
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error fetching exchange rate. Using fallback. Error: {e}")
        st.session_state['exchange_rate_source'] = 'Fallback (Network Error)'
        return 83.50
    except json.JSONDecodeError:
        st.warning("Invalid response from exchange rate API. Using fallback rate.")
        st.session_state['exchange_rate_source'] = 'Fallback (Invalid Response)'
        return 83.50

def load_tax_rules(file_path="duty_taxes.csv"):
    """
    Loads tax rules using Pathway in static mode. This is NOT cached to ensure
    the latest data from the CSV is read on every script run.
    """
    if not os.path.exists(file_path):
        st.error(f"FATAL ERROR: `{file_path}` not found. Please ensure it is in the same directory.")
        return None, None
    try:
        last_modified_time = os.path.getmtime(file_path)
        tax_rules_table = pw.io.csv.read(file_path, schema=TaxRule, mode="static")
        df = debug.table_to_pandas(tax_rules_table)
        return df, last_modified_time
    except Exception as e:
        st.error(f"Error loading or parsing `duty_taxes.csv`: {e}")
        return None, None

# --- REFACTORED: API configuration is no longer done inside these functions ---
def extract_query_details(user_query: str) -> QueryDetails | None:
    """Uses Gemini to extract structured data from the user's query."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = textwrap.dedent(f"""
        From the user's query, extract the product, origin country, price in USD, and destination country.
        The destination country defaults to 'India' if not specified.
        Respond ONLY with a valid JSON object that matches this schema:
        {QueryDetails.model_json_schema()}

        User Query: "{user_query}"
        """)
        response = model.generate_content(prompt)
        raw_text = response.text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            st.error("AI Error: Could not find a valid JSON object in the response.")
            st.write("AI Raw Output:", raw_text)
            return None
        json_str = match.group(0)
        data = json.loads(json_str)
        return QueryDetails(**data)
    except Exception as e:
        st.error(f"An error occurred while communicating with the AI for data extraction: {e}")
        return None

def generate_ai_summary(details: dict) -> str:
    """Generates the final human-friendly summary using Gemini."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = textwrap.dedent(f"""
        You are TaxWise AI, a helpful assistant for calculating import costs.
        A user asked about importing a {details['product']} from {details['country']} to {details['destination']}.
        The final, all-inclusive price is ₹{details['total_inr']:,.2f} INR.
        This was calculated from a base price of ${details['base_price_usd']:,.2f} USD, with total taxes and fees of ${details['tax_usd']:,.2f} USD.
        Explain this breakdown in 2-3 friendly sentences. Start by stating the final price boldly with markdown. Emphasize that this is the true, landed cost with no hidden fees. Use an emoji. 🚚
        """)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"AI summary generation failed: {e}. Using a fallback summary.")
        return f"The final estimated cost for your {details['product']} is **₹{details['total_inr']:,.2f} INR**. This includes the product's base price of ${details['base_price_usd']:,.2f} plus all relevant taxes and fees of ${details['tax_usd']:,.2f}."

def normalize(x: str) -> str:
    """Standardizes strings for reliable matching."""
    return str(x).lower().strip().replace("united states", "usa").replace("laptops", "laptop")

# --- 4. Main App Interface ---

# Load data once per script rerun for the sidebar display. This keeps the sidebar
# "live" as any user interaction causes a rerun.
tax_rules_df_for_display, last_mod_time = load_tax_rules()
if tax_rules_df_for_display is not None:
    st.sidebar.subheader("Live Data Source (`duty_taxes.csv`)")
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_mod_time))
    st.sidebar.caption(f"Last updated: {ts}")
    with st.sidebar.expander("Click to view raw data"):
        st.dataframe(tax_rules_df_for_display)

st.title("💡 TaxWise AI")
st.markdown("Your real-time guide to transparent import pricing. Get the true final cost, with all taxes and duties included.")
st.divider()

st.subheader("What are you importing?")
user_question = st.text_input(
    "Enter a description:",
    placeholder="e.g., A $1300 laptop from USA to India",
    label_visibility="collapsed"
)

if st.button("Calculate True Cost", type="primary", use_container_width=True):
    # --- REFACTORED: Centralized validation and processing flow ---

    # 1. Validate all inputs first
    if not gemini_api_key:
        st.error("Please enter your Gemini API Key in the sidebar to use the app.")
        st.stop()
    if not user_question:
        st.warning("Please ask a question.")
        st.stop()

    tax_rules_df, _ = load_tax_rules()
    if tax_rules_df is None:
        st.error("Could not load tax rules. Please check the `duty_taxes.csv` file.")
        st.stop()

    # 2. Configure the API (just-in-time)
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.error(f"Failed to configure Gemini API. Please check your key. Error: {e}")
        st.stop()

    # 3. Start the processing pipeline
    extracted_data = None
    with st.spinner("🤖 AI is analyzing your question..."):
        extracted_data = extract_query_details(user_question)

    if extracted_data:
        with st.spinner("🧮 Finding tax rules and calculating costs..."):
            product_norm = normalize(extracted_data.product)
            origin_norm = normalize(extracted_data.country)
            dest_norm = normalize(extracted_data.destination)

            tax_rules_df['product_norm'] = tax_rules_df['product_category'].apply(normalize)
            tax_rules_df['origin_norm'] = tax_rules_df['origin_country'].apply(normalize)
            tax_rules_df['dest_norm'] = tax_rules_df['destination_country'].apply(normalize)

            match = tax_rules_df[
                (tax_rules_df['product_norm'] == product_norm) &
                (tax_rules_df['origin_norm'] == origin_norm) &
                (tax_rules_df['dest_norm'] == dest_norm)
            ]

        if not match.empty:
            rule = match.iloc[0]
            exchange_rate_inr = get_live_exchange_rate_inr(exchange_rate_api_key)
            duty_tax_usd = (extracted_data.price_usd * rule['tax_rate']) + rule['fixed_fee_usd']
            total_cost_usd = extracted_data.price_usd + duty_tax_usd
            total_cost_inr = total_cost_usd * exchange_rate_inr
            details = {
                "product": extracted_data.product, "country": extracted_data.country,
                "destination": extracted_data.destination, "base_price_usd": extracted_data.price_usd,
                "tax_usd": duty_tax_usd, "total_inr": total_cost_inr
            }

            st.subheader("✅ Calculation Complete!")
            with st.spinner("✍️ AI is writing your summary..."):
                ai_summary = generate_ai_summary(details)
            st.markdown(ai_summary)
            st.divider()
            
            col1, col2 = st.columns([2, 3])
            with col1:
                st.metric("True Final Cost (INR)", f"₹{total_cost_inr:,.2f}")
            with col2:
                rate_source = st.session_state.get('exchange_rate_source', 'N/A')
                st.markdown(f"**Cost Breakdown** (1 USD = ₹{exchange_rate_inr:,.2f})")
                st.markdown(f"via `{rate_source}`")
                st.markdown(f"""
                - **Base Price:** `${extracted_data.price_usd:,.2f}`
                - **Import Duties & Fees:** `${duty_tax_usd:,.2f}`
                - **Total (USD):** `${total_cost_usd:,.2f}`
                """)
        else:
            st.error(
                f"Sorry, I couldn't find a tax rule for a '{extracted_data.product}' from "
                f"'{extracted_data.country}' to '{extracted_data.destination}'. "
                "Please check the data source in the sidebar or try another combination."
            )

