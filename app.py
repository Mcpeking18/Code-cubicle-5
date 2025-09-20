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
from typing import List, Optional
from dotenv import load_dotenv

# --- 1. Setup & Configuration ---
st.set_page_config(
    page_title="TaxWise AI",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
gemini_api_key = os.getenv("gemini_api_key")
exchange_rate_api_key = os.getenv("exchange_rate_api_key")

# --- Sidebar for Configuration Status ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("API keys are loaded from the `.env` file.")
    if gemini_api_key:
        st.success("Gemini API Key: Configured ✅")
    else:
        st.error("Gemini API Key: Not found in `.env`. Please add `GEMINI_API_KEY` to the `.env` file.")
    if exchange_rate_api_key:
        st.success("ExchangeRate-API Key: Configured ✅")
    else:
        st.warning("ExchangeRate-API Key: Not found in `.env`. Using free tier API as fallback.")

# --- 2. Pydantic Schemas ---
class QueryDetails(BaseModel):
    """Structured representation of the user's import query."""
    product: str = Field(..., description="The product category, e.g., 'laptop', 'smartphone'")
    country: str = Field(..., description="The country of origin, e.g., 'USA', 'Japan'")
    price_usd: float = Field(..., description="The price of the product in USD")

# --- 3. Core Functions ---
@st.cache_data(ttl=3600)  # Cache the exchange rate for 1 hour
def get_live_exchange_rate(api_key=None, target_currency="INR"):
    """
    Fetches the live USD to a target currency exchange rate with a reliable fallback.
    Handles both v4 and v6 API responses correctly.
    """
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

        # Handle both 'rates' (v4) and 'conversion_rates' (v6)
        rates_dict = data.get("conversion_rates") or data.get("rates")

        if data.get("result") == "success" and rates_dict and target_currency in rates_dict:
            return rates_dict[target_currency]
        else:
            error_details = data.get("error-type", f"API response did not contain valid rates for {target_currency}.")
            st.warning(f"Exchange rate API error: '{error_details}'. Using fallback rate.")
            st.session_state['exchange_rate_source'] = 'Fallback (API Error)'
            return 83.50  # Fallback for INR
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error fetching exchange rate: {e}. Using fallback rate.")
        st.session_state['exchange_rate_source'] = 'Fallback (Network Error)'
        return 83.50
    except json.JSONDecodeError:
        st.warning("Invalid JSON response from exchange rate API. Using fallback rate.")
        st.session_state['exchange_rate_source'] = 'Fallback (Invalid Response)'
        return 83.50

@st.cache_data
def load_tax_rules(file_path="duty_taxes.csv"):
    """
    Loads tax rules using pandas for reliable caching and returns the DataFrame,
    last modified time, and a list of unique destinations.
    """
    if not os.path.exists(file_path):
        st.error(f"FATAL ERROR: `{file_path}` not found. Please ensure it's in the same directory.")
        return None, None, []
    try:
        last_modified_time = os.path.getmtime(file_path)
        df = pd.read_csv(file_path)
        unique_destinations = sorted(df['destination_country'].str.strip().unique().tolist())
        return df, last_modified_time, unique_destinations
    except Exception as e:
        st.error(f"Error loading or parsing `{file_path}`: {e}")
        return None, None, []

def extract_query_details(user_query: str) -> Optional[QueryDetails]:
    """Uses Gemini to extract structured data from the user's query."""
    if not gemini_api_key:
        st.error("Gemini API key not configured in `.env`. Please add `GEMINI_API_KEY`.")
        return None
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        schema_json = json.dumps(QueryDetails.model_json_schema(), indent=2)

        prompt = textwrap.dedent(f"""
            From the user's query, extract the product, origin country, and price in USD.
            Respond ONLY with a valid JSON object that strictly follows this JSON schema:
            {schema_json}

            User Query: "{user_query}"
            """)
        response = model.generate_content(prompt)
        raw_text = response.text
        match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", raw_text, re.DOTALL)
        if not match:
            st.error("AI Error: Could not find a valid JSON object in the response.")
            st.write("AI Raw Output:", raw_text)
            return None

        json_str = match.group(1) or match.group(2)
        data = json.loads(json_str)
        return QueryDetails(**data)
    except Exception as e:
        st.error(f"Error communicating with the AI for data extraction: {e}")
        return None

def generate_ai_summary(details: dict, comparison_df: pd.DataFrame) -> str:
    """Generates a human-friendly summary using Gemini, now including comparison results."""
    if not gemini_api_key:
        return "Gemini API key not configured in `.env`."
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        best_deal = comparison_df.sort_values(by="Total Cost (USD)").iloc[0]
        
        prompt = textwrap.dedent(f"""
            You are TaxWise AI, a helpful assistant for calculating import costs.
            A user asked about importing a {details['product']} from {details['country']} (base price ${details['base_price_usd']:,.2f}).
            We calculated the costs for multiple destinations. The best deal is importing to **{best_deal['Destination']}**, with a final landed cost of **{best_deal['Final Cost']}**.

            - Start with a bold markdown sentence stating the best option found.
            - Briefly explain that this final price is the true, all-inclusive "landed cost."
            - Mention that the table below provides a full breakdown for all selected countries.
            - Use a friendly and encouraging tone with an emoji. 🌎
            """)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"AI summary generation failed: {e}. Using a fallback summary.")
        best_deal = comparison_df.sort_values(by="Total Cost (USD)").iloc[0]
        return f"The best option found is importing to **{best_deal['Destination']}** with a final estimated cost of **{best_deal['Final Cost']}**. This includes the product's base price of ${details['base_price_usd']:,.2f} plus all relevant taxes and fees. See the table below for a full comparison."

def normalize(x: str) -> str:
    """Standardizes strings for reliable matching."""
    return str(x).lower().strip().replace("united states", "usa").replace("laptops", "laptop")

# --- 4. Main App Interface ---
st.title("💡 TaxWise AI")
st.markdown("### Your real-time guide to transparent import pricing. ✈️")
st.markdown("Instantly compare the true final cost of a product across different countries, with all taxes and duties included.")
st.divider()

# --- Load data once and display in sidebar ---
tax_rules_df, last_mod_time, destination_options = load_tax_rules()
if tax_rules_df is not None:
    with st.sidebar:
        st.subheader("Live Data Source (`duty_taxes.csv`)")
        if last_mod_time:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_mod_time))
            st.caption(f"Last updated: {ts}")
        with st.expander("Click to view raw data"):
            st.dataframe(tax_rules_df)

# --- User Input Section ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. What are you importing?")
    user_question = st.text_input(
        "Enter a description:",
        placeholder="e.g., A $1300 laptop from the USA",
        label_visibility="collapsed"
    )
with col2:
    st.subheader("2. Where might you ship it?")
    selected_destinations = st.multiselect(
        "Select one or more destination countries:",
        options=destination_options,
        default=["India", "Germany"] if "India" in destination_options and "Germany" in destination_options else [],
        label_visibility="collapsed"
    )

if st.button("Calculate & Compare True Cost", type="primary", use_container_width=True):
    # 1. Validate all inputs
    if not gemini_api_key:
        st.error("Please configure `GEMINI_API_KEY` in the `.env` file to use the app.")
        st.stop()
    if not user_question:
        st.warning("Please describe the item you are importing.")
        st.stop()
    if not selected_destinations:
        st.warning("Please select at least one destination country.")
        st.stop()
    if tax_rules_df is None:
        st.error("Could not load tax rules. Please check the `duty_taxes.csv` file.")
        st.stop()

    # 2. Start the processing pipeline
    extracted_data = None
    with st.spinner("🤖 AI is analyzing your question..."):
        extracted_data = extract_query_details(user_query=user_question)

    if extracted_data:
        results = []
        with st.spinner("🧮 Calculating costs for each destination..."):
            product_norm = normalize(extracted_data.product)
            origin_norm = normalize(extracted_data.country)

            # Pre-normalize the DataFrame for efficiency
            tax_rules_df['product_norm'] = tax_rules_df['product_category'].apply(normalize)
            tax_rules_df['origin_norm'] = tax_rules_df['origin_country'].apply(normalize)
            tax_rules_df['dest_norm'] = tax_rules_df['destination_country'].apply(normalize)

            for destination in selected_destinations:
                dest_norm = normalize(destination)
                match = tax_rules_df[
                    (tax_rules_df['product_norm'] == product_norm) &
                    (tax_rules_df['origin_norm'] == origin_norm) &
                    (tax_rules_df['dest_norm'] == dest_norm)
                ]

                if not match.empty:
                    rule = match.iloc[0]
                    exchange_rate = get_live_exchange_rate(exchange_rate_api_key, "INR")

                    duty_tax_usd = (extracted_data.price_usd * rule['tax_rate']) + rule['fixed_fee_usd']
                    total_cost_usd = extracted_data.price_usd + duty_tax_usd
                    total_cost_local = total_cost_usd * exchange_rate

                    results.append({
                        "Destination": destination,
                        "Base Price (USD)": extracted_data.price_usd,
                        "Taxes & Fees (USD)": duty_tax_usd,
                        "Total Cost (USD)": total_cost_usd,
                        "Final Cost": f"₹{total_cost_local:,.2f}"
                    })
                else:
                    results.append({
                        "Destination": destination,
                        "Base Price (USD)": extracted_data.price_usd,
                        "Taxes & Fees (USD)": "N/A",
                        "Total Cost (USD)": "N/A",
                        "Final Cost": "No Rule Found"
                    })

        if results:
            st.divider()
            st.subheader("✅ Calculation Complete!")

            comparison_df = pd.DataFrame(results)
            comparison_df_sortable = comparison_df.copy()
            comparison_df_sortable['sort_col'] = pd.to_numeric(comparison_df_sortable['Total Cost (USD)'], errors='coerce')

            # --- AI Summary ---
            details = {
                "product": extracted_data.product, "country": extracted_data.country,
                "base_price_usd": extracted_data.price_usd
            }
            if not comparison_df_sortable['sort_col'].dropna().empty:
                with st.spinner("✍️ AI is writing your summary..."):
                    ai_summary = generate_ai_summary(details, comparison_df_sortable)
                st.markdown(ai_summary)

            # --- Display Results Table ---
            st.markdown(f"**Comparison Breakdown** (1 USD ≈ ₹{get_live_exchange_rate(exchange_rate_api_key, 'INR'):.2f} INR)")
            rate_source = st.session_state.get('exchange_rate_source', 'N/A')
            st.caption(f"Currency rate via `{rate_source}`")
            
            def highlight_min(s):
                is_min = s == s.min()
                return ['background-color: #28a745; color: white' if v else '' for v in is_min]

            display_df = comparison_df_sortable.drop(columns=['sort_col']).set_index("Destination")
            
            if not comparison_df_sortable['sort_col'].dropna().empty:
                st.dataframe(
                    display_df.style.format({
                        "Base Price (USD)": "${:,.2f}",
                        "Taxes & Fees (USD)": "${:,.2f}",
                        "Total Cost (USD)": "${:,.2f}",
                    }).apply(highlight_min, subset=['Total Cost (USD)']),
                    use_container_width=True
                )
            else:
                st.dataframe(display_df, use_container_width=True)
        else:
            st.error(f"Sorry, I couldn't find a tax rule for the requested item. Please check the data source or try another combination.")