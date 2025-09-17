# AI Money Loop System

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your API keys.

## Running
- **Streamlit Dashboard:**
  ```
  streamlit run main.py
  ```
- **Recurring Automated Simulation:**
  ```
  python main.py
  ```
  Uncomment `run_recurring()` in main.py.

## Logs
All logs are written to `ai_money_loop.log`.
