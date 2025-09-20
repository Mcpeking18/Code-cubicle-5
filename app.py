import streamlit as st
import requests
import pathway as pw
from pathway import debug
import os
import pandas as pd
import google.generativeai as genai
import json
from pydantic import BaseModel, Field
import re

# --- 1. Setup & Configuration ---
st.set_page_config(page_title="TaxWise AI", page_icon="💡", layout="centered")

# Configure Gemini API key from environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# --- 2. Pydantic Models for Data Extraction ---
class QueryDetails(BaseModel):
    product: str = Field(..., description="The product category, e.g., 'laptop', 'smartphone'")
    country: str = Field(..., description="The country of origin, e.g., 'USA', 'Japan'")
    price_usd: float = Field(..., description="The price of the product in USD")

# --- 3. Core Functions ---
@st.cache_data(ttl=300)
def get_live_exchange_rate():
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
        response.raise_for_status()
        return response.json()["rates"]["INR"]
    except requests.exceptions.RequestException:
        return 83.50

def extract_entities_with_gemini(user_query: str) -> QueryDetails | None:
    """Uses Gemini to extract structured data from a user's question."""
    if not gemini_api_key: return None
    
    # CORRECTED MODEL NAME
    model = genai.GenerativeModel("gemini-2.0-flash") 
    prompt = f"""
    From the following user query, extract the product, country of origin, and price in USD.
    Respond ONLY with a valid JSON object that matches this schema:
    {QueryDetails.model_json_schema()}

    User Query: "{user_query}"
    """
    
    try:
        response = model.generate_content(prompt)
        # Use regex to find the JSON block within the AI's response
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return QueryDetails(**data)
        else:
            st.error("The AI responded, but I couldn't find valid JSON data in its answer.")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI extraction: {e}")
        return None

def get_gemini_summary(details: dict) -> str:
    """Generate a natural-language summary of the calculation with Gemini."""
    if not gemini_api_key:
        return "Gemini API key missing - cannot generate AI summary."

    # Get currency symbol
    currency_symbols = {
        "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", 
        "JPY": "¥", "CAD": "C$", "AUD": "A$", "CNY": "¥"
    }
    target_symbol = currency_symbols.get(details['target_currency'], details['target_currency'])
    
    # Create a fallback summary in case Gemini fails
    fallback_summary = f"Final cost is {target_symbol}{details['total_converted']:,.2f}. This includes the product price, taxes, and fees."
    
    try:
        # Use a more reliable model version
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""
        You are TaxWise AI. A user is importing a {details['product']} from {details['country']}.
        Breakdown:
        Base = ${details['base_price_usd']:,.2f}, 
        Taxes = ${details['tax_usd']:,.2f}, 
        Total = ${details['total_usd']:,.2f} USD 
        → Final in {details['target_currency']} = {target_symbol}{details['total_converted']:,.2f} at {details['exchange_rate']:.2f}/{details['base_currency']}.

        Write 2 short, friendly sentences starting with the final {details['target_currency']} cost. 
        Reassure the user this includes everything (no hidden fees).
        
        If the tax rate is unusually high (>40%), add a warning about potential scams or suggest verifying with official sources.
        """
        
        response = model.generate_content(prompt, timeout=5)  # Add timeout
        if response and hasattr(response, 'text'):
            return response.text
        return fallback_summary
    except Exception as e:
        st.error(f"Error generating AI summary: {str(e)}")
        return fallback_summary

# --- 3. Data Loading ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "duty_taxes.csv")
if not os.path.exists(file_path):
    st.error("FATAL ERROR: `duty_taxes.csv` not found.")
    st.stop()

class TaxRule(pw.Schema):
    product_category: str; origin_country: str; tax_rate: float; fixed_fee_usd: float
tax_rules_table = pw.io.csv.read(file_path, schema=TaxRule, mode="streaming", autocommit_duration_ms=50)

# --- 4. Streamlit User Interface ---
st.title("💡 TaxWise AI")
st.markdown("Your real-time guide to transparent import pricing.")
st.divider()

with st.container(border=True):
    st.subheader("Why This App Matters")
    st.markdown("When people shop abroad, hidden taxes and duties create confusion. **TaxWise AI provides the true, final price from a simple question.**")
st.divider()

st.subheader("Ask Your Question")
user_question = st.text_area(
    "Enter your query here...",
    placeholder="e.g., What is the total cost for a $1200 laptop from the USA in INR?",
    height=100
)

if st.button("Calculate True Final Cost", type="primary", use_container_width=True):
    if not user_question:
        st.warning("Please ask a question.")
    else:
        with st.spinner("🤖 AI is analyzing your question..."):
            extracted_data = extract_entities_with_gemini(user_question)

        if extracted_data:
            live_rules_df = debug.table_to_pandas(tax_rules_table)
            match = live_rules_df[
                (live_rules_df['product_category'].str.lower() == extracted_data.product.lower()) &
                (live_rules_df['origin_country'].str.lower() == extracted_data.country.lower())
            ]
            
            if not match.empty:
                exchange_rate_inr = get_live_exchange_rate()
                rule = match.iloc[0]
                tax_rate, fixed_fee = rule['tax_rate'], rule['fixed_fee_usd']

                duty_tax_usd = (extracted_data.price_usd * tax_rate) + fixed_fee
                total_cost_usd = extracted_data.price_usd + duty_tax_usd
                total_cost_inr = total_cost_usd * exchange_rate_inr
                
                calculation_details = {
                    "product": extracted_data.product, "country": extracted_data.country,
                    "base_price_usd": extracted_data.price_usd, "tax_usd": duty_tax_usd,
                    "tax_rate": tax_rate, "total_usd": total_cost_usd,
                    "total_inr": total_cost_inr, "exchange_rate": exchange_rate_inr,
                }

                with st.spinner("🤖 Calculating and generating final summary..."):
                    summary = get_gemini_summary(calculation_details)

                st.subheader("💡 AI-Powered Answer")
                st.markdown(summary)
                st.divider()
                st.subheader("💰 Price Breakdown")
                st.metric("True Final Cost (INR)", f"₹{total_cost_inr:,.2f}")
                
                col_br1, col_br2 = st.columns(2)
                with col_br1: st.metric("Base Price (USD)", f"${extracted_data.price_usd:,.2f}")
                with col_br2: st.metric("Taxes & Fees (USD)", f"${duty_tax_usd:,.2f}")
                
                st.info(f"Calculation based on a real-time exchange rate of 1 USD = ₹{exchange_rate_inr:.2f} INR.", icon="💹")
            else:
                st.error(f"⚠️ No tax rule found for a '{extracted_data.product}' from '{extracted_data.country}'. Please check your spelling or our database.")