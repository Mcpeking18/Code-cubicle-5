# 💡 TaxWise: Real-Time AI Import Cost Calculator

TaxWise is an intelligent, real-time application built for the **Code Cubicle 5.0 Hackathon**.  
It demystifies the true cost of importing products by instantly calculating the final **landed cost**, including complex duty taxes and live currency conversions — all from a simple, natural language query.

---
## 🛑 The Problem

When buying products from other countries, the sticker price is misleading.  
Consumers are often surprised by hidden costs from:

- Complex and stale duty tax information  
- Fluctuating currency exchange rates  
- A lack of easy-to-use, transparent pricing tools  

---

## ✅ Our Solution

TaxWise provides a seamless and accurate solution with three key features:

1. **Natural Language Query**  
   Ask questions like:  
   > *"a 130,000 won South Korean smartphone to Germany"*  
   Our AI, powered by **Google Gemini**, instantly understands the product, origin, and price.

2. **Real-Time Data Pipeline**  
   We use **Pathway** to create a streaming data pipeline that continuously monitors our tax and currency databases.  
   If a rule changes, our app knows instantly — ensuring calculations are always up-to-date.

3. **Multi-Currency Conversion**  
   The app automatically handles conversions:  
   - From the product's original currency → **USD** (for calculations)  
   - Then into the **destination country’s local currency** for a clear final price  

---

## 🏗️ Tech Stack & Architecture

The application runs as two distinct microservices, orchestrated by **Docker Compose**, ensuring a scalable and robust architecture:

- **pathway-engine (Backend)**  
  A Python service running a **Pathway** pipeline.  
  It watches source CSV files for changes, enriches the data, and writes live results to a JSON file.

- **streamlit-app (Frontend)**  
  A **Streamlit** web server that reads live JSON data from the engine.  
  It handles user interaction, calls the **Gemini API** for NLP, and displays final calculations.

---

## 🚀 Getting Started

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and **Docker Compose** must be installed  
- API keys for **Google Gemini** and **ExchangeRate-API**

---

### 1️⃣ Clone the Repository

```bash
git clone [repo-url]
cd [repo-name]
````

---

### 2️⃣ Configure Environment Variables

Create a `.env` file in the project’s root directory.
Copy the contents below and add your secret API keys:

```env
gemini_api_key="PASTE_YOUR_GEMINI_API_KEY_HERE"
exchange_rate_api_key="PASTE_YOUR_EXCHANGERATE_API_KEY_HERE"
```

---

### 3️⃣ Run the Application

Build the Docker images and launch both services with:

```bash
docker-compose up --build
```

The terminal will show logs from both the Pathway engine and the Streamlit app.

---

### 4️⃣ Access TaxWise

Once the containers are running, open your browser and go to:

👉 [http://localhost:8501](http://localhost:8501)

---

## 🔄 See Real-Time in Action!

1. While the application is running, open:

   * `duty_taxes.csv`

2. Change any value (e.g., a tax rate) and save the file.

3. The **Pathway engine** will instantly detect the change and update its output.

4. The **next calculation** you run in the app will use the updated data! 🎉

---

## 👨‍💻 Contributors

Built with ❤️ by **Team Code Blooded** for Code Cubicle 5.0.

* Rishi Verma 
* Pramod Mohanty 
* Riya Singh 
* Vanshika Chawla

It is still a prototype <3
