# 💡 TaxWise

**TaxWise AI** is a real-time AI application built for the **Code Cubicle 5.0 Hackathon** (Track 2: Stale AI). It solves the problem of confusing and hidden import costs by allowing users to ask natural language questions and get the true, all-inclusive price of a product, including live duty taxes and up-to-the-second currency conversions.

---

## Core Technologies

* **Pathway:** For creating a real-time, streaming data pipeline that continuously monitors the duty tax database.
* **Google Gemini:** Used for both understanding the user's natural language query and generating a human-friendly summary of the final cost.
* **Streamlit:** For building the interactive web user interface.
* **Docker:** To containerize the application, ensuring it runs consistently in any environment.

---

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Create Your Data File**
    Create a `duty_taxes.csv` file in the root directory with the following headers:
    ```csv
    product_category,origin_country,tax_rate,fixed_fee_usd
    laptop,USA,0.18,25
    smartphone,China,0.25,10
    ```

3.  **Build the Docker Image**
    ```bash
    docker build -t taxwise-ai .
    ```

4.  **Run the Container**
    Replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key.
    ```bash
    docker run -p 8501:8501 -v .:/app -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE" taxwise-ai
    ```

5.  **Access the App**
    Open your browser and navigate to `http://localhost:8501`.
