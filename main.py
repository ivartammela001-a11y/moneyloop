# AI Money Loop: Complete Project Template

import openai
import requests
import random
from datetime import datetime, timedelta
import time
import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os

# -------------------------
# Load Environment Variables
# -------------------------
try:
    load_dotenv()
except:
    pass  # Continue without .env file

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'demo_key')
CANVA_API_KEY = os.getenv('CANVA_API_KEY', 'demo_key')
ETSY_API_KEY = os.getenv('ETSY_API_KEY', 'demo_key')
GUMROAD_API_KEY = os.getenv('GUMROAD_API_KEY', 'demo_key')

if OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key':
    openai.api_key = OPENAI_API_KEY

# -------------------------
# Logging
# -------------------------
logging.basicConfig(filename='ai_money_loop.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# -------------------------
# Data Structures
# -------------------------
investment_tracker = [
    {'layer':1, 'starting_capital':50, 'expected_profit':250, 'actual_profit':0, 'platform':'Etsy'},
    {'layer':2, 'starting_capital':250, 'expected_profit':1000, 'actual_profit':0, 'platform':'Gumroad'},
    {'layer':3, 'starting_capital':1000, 'expected_profit':3000, 'actual_profit':0, 'platform':'Shopify/Printful'}
]

product_ideas = [
    {'niche':'Motivational', 'product':'A4 Inspirational Quote Printable', 'status':'Listed'},
    {'niche':'Fitness', 'product':'7-Day Meal Planner Template', 'status':'Listed'},
    {'niche':'Productivity', 'product':'Daily Planner Template', 'status':'In Production'}
]

profit_reinvestment = [
    {'current_capital':50,'suggested_layer':'Layer 1','action':'Create 5 digital printables'},
    {'current_capital':250,'suggested_layer':'Layer 2','action':'Scale templates & e-books'},
    {'current_capital':1000,'suggested_layer':'Layer 3','action':'Launch POD store'}
]

next_layer = lambda capital: 'Layer 1' if capital<250 else ('Layer 2' if capital<1000 else 'Layer 3')

# -------------------------
# API Integration Functions
# -------------------------

def generate_ai_text(prompt):
    if OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key':
        try:
            response = openai.ChatCompletion.create(model="gpt-4", prompt=prompt, max_tokens=500)
            return response['choices'][0]['message']['content']
        except:
            pass
    # Demo mode - return sample content
    return f"AI Generated Content for: {prompt} - This is a demo simulation!"

def generate_ai_image(prompt):
    if OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key':
        try:
            response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
            return response['data'][0]['url']
        except:
            pass
    # Demo mode - return placeholder
    return "https://via.placeholder.com/1024x1024/4CAF50/FFFFFF?text=AI+Generated+Image"

def export_to_platform(product_name, content_file, image_file, platform):
    logging.info(f"Uploading {product_name} to {platform}")
    # Replace with real API requests to Etsy/Gumroad/Shopify/Canva
    return True

# -------------------------
# Automation Functions
# -------------------------

def simulate_layer(layer):
    logging.info(f"Simulating Layer {layer['layer']} on {layer['platform']}")
    for product in product_ideas:
        try:
            text = generate_ai_text(product['niche'] + ' ' + product['product'])
            image_url = generate_ai_image(product['product'])
            export_to_platform(product['product'], text, image_url, layer['platform'])
        except Exception as e:
            logging.error(f"Error generating/exporting product {product['product']}: {e}")
    profit = layer['starting_capital'] * random.uniform(0.5, 1.5)
    layer['actual_profit'] = round(profit,2)
    logging.info(f"Layer {layer['layer']} profit: €{layer['actual_profit']}")
    return layer['actual_profit']

def suggest_next_action(capital):
    layer_name = next_layer(capital)
    logging.info(f"Next Layer: {layer_name}")
    for item in profit_reinvestment:
        if item['suggested_layer'] == layer_name:
            logging.info(f"Action: {item['action']}")

# -------------------------
# Streamlit Dashboard
# -------------------------

def run_dashboard():
    global OPENAI_API_KEY
    
    st.title('💰 AI Money Loop Dashboard')
    st.markdown("---")
    
    # Check if API keys are already set
    api_keys_set = OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key'
    
    # Welcome message for first-time users
    if not api_keys_set:
        st.info("🎯 **Welcome to AI Money Loop!** This system automates digital product creation and sales across multiple platforms. Configure your API keys in the sidebar to unlock full AI functionality.")
    
    # API Keys Configuration Section
    st.sidebar.header('🔑 API Configuration')
    
    if not api_keys_set:
        st.sidebar.warning('⚠️ No API keys detected. Enter your keys below for full functionality.')
        
        with st.sidebar.expander("🔧 Configure API Keys", expanded=True):
            openai_key = st.text_input(
                "OpenAI API Key", 
                value="",
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys",
                placeholder="sk-proj-..."
            )
            
            if st.button("💾 Save API Keys"):
                if openai_key:
                    # Update the global variable
                    OPENAI_API_KEY = openai_key
                    openai.api_key = openai_key
                    st.sidebar.success("✅ OpenAI API Key saved!")
                    st.rerun()
                else:
                    st.sidebar.error("Please enter a valid OpenAI API Key")
    else:
        st.sidebar.success("✅ API Keys configured!")
        if st.sidebar.button("🔄 Reset API Keys"):
            OPENAI_API_KEY = 'demo_key'
            openai.api_key = None
            st.rerun()
    
    # Main Dashboard Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button('🚀 Simulate Sales', type="primary"):
            for layer in investment_tracker:
                profit = simulate_layer(layer)
                suggest_next_action(profit)
            st.success('Sales simulation complete!')
    
    with col2:
        if api_keys_set:
            st.success("🤖 AI Mode: Real API calls")
        else:
            st.info("🎭 Demo Mode: Simulated responses")

    df_tracker = pd.DataFrame(investment_tracker)
    df_tracker['ROI (%)'] = ((df_tracker['actual_profit'] - df_tracker['starting_capital']) / df_tracker['starting_capital'] * 100).round(2)

    st.subheader('Investment Tracker')
    st.dataframe(df_tracker)

    st.subheader('ROI per Layer')
    st.bar_chart(df_tracker[['layer','ROI (%)']].set_index('layer'))

    current_capital = df_tracker['actual_profit'].max()
    layer_name, action = next_layer(current_capital), ''
    for item in profit_reinvestment:
        if item['suggested_layer'] == next_layer(current_capital):
            action = item['action']
    st.subheader('Next Layer Actions')
    st.write(f'Current Capital: €{current_capital}')
    st.write(f'Next Layer: {layer_name}, Suggested Action: {action}')

# -------------------------
# Recurring Scheduler
# -------------------------

def run_recurring(simulation_interval_hours=24):
    capital = 50
    while True:
        for layer in investment_tracker:
            profit = simulate_layer(layer)
            capital = profit
            suggest_next_action(capital)
        logging.info(f"End of simulation round. Current capital: €{capital}")
        time.sleep(simulation_interval_hours * 3600)

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    # Uncomment one of the two options below:
    # Option 1: Run Streamlit Dashboard
    run_dashboard()

    # Option 2: Run recurring simulation in production
    # run_recurring(simulation_interval_hours=24)
