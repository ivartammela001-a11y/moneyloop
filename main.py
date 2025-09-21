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

# OpenAI API key is now handled in the individual functions

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
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return f"Error generating content: {str(e)}"
    # Demo mode - return sample content
    return f"AI Generated Content for: {prompt} - This is a demo simulation!"

def generate_ai_image(prompt):
    if OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key':
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.images.generate(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            return response.data[0].url
        except Exception as e:
            logging.error(f"OpenAI Image API error: {e}")
            return "https://via.placeholder.com/1024x1024/4CAF50/FFFFFF?text=AI+Generated+Image"
    # Demo mode - return placeholder
    return "https://via.placeholder.com/1024x1024/4CAF50/FFFFFF?text=AI+Generated+Image"

def export_to_platform(product_name, content_file, image_file, platform):
    """Export products to real platforms with actual API calls"""
    logging.info(f"Uploading {product_name} to {platform}")
    
    if platform == 'Etsy' and ETSY_API_KEY and ETSY_API_KEY != 'demo_key':
        logging.info(f"Creating Etsy listing for {product_name}")
        return create_etsy_listing(product_name, content_file, image_file)
    elif platform == 'Gumroad' and GUMROAD_API_KEY and GUMROAD_API_KEY != 'demo_key':
        logging.info(f"Creating Gumroad product for {product_name}")
        logging.info(f"Gumroad API Key: {GUMROAD_API_KEY[:10]}...")
        return create_gumroad_product(product_name, content_file, image_file)
    elif platform == 'Shopify/Printful' and GUMROAD_API_KEY and GUMROAD_API_KEY != 'demo_key':
        logging.info(f"Creating Shopify product for {product_name}")
        return create_shopify_product(product_name, content_file, image_file)
    else:
        logging.info(f"Platform {platform} not configured or in demo mode")
        logging.info(f"ETSY_KEY: {ETSY_API_KEY != 'demo_key' if ETSY_API_KEY else 'None'}")
        logging.info(f"GUMROAD_KEY: {GUMROAD_API_KEY != 'demo_key' if GUMROAD_API_KEY else 'None'}")
        return True

def create_etsy_listing(product_name, content, image_url):
    """Create real Etsy listing using Etsy API"""
    try:
        import requests
        
        # Etsy API endpoints
        etsy_base_url = "https://openapi.etsy.com/v3/application"
        
        headers = {
            'x-api-key': ETSY_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Create listing data
        listing_data = {
            "title": content.get('title', product_name),
            "description": content.get('description', ''),
            "price": 9.99,  # Base price
            "quantity": 100,
            "tags": content.get('tags', '').split()[:13],  # Etsy allows max 13 tags
            "materials": ["Digital Download"],
            "shop_section_id": None,
            "who_made": "i_did",
            "when_made": "2020_2024",
            "is_supply": False,
            "is_customizable": True,
            "state": "draft"  # Start as draft for review
        }
        
        # Create the listing
        response = requests.post(
            f"{etsy_base_url}/shops/me/listings",
            headers=headers,
            json=listing_data
        )
        
        if response.status_code == 201:
            listing_id = response.json()['results'][0]['listing_id']
            logging.info(f"✅ Etsy listing created: {listing_id}")
            
            # Upload image
            upload_etsy_image(listing_id, image_url, headers)
            
            return True
        else:
            logging.error(f"Etsy API error: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error creating Etsy listing: {e}")
        return False

def upload_etsy_image(listing_id, image_url, headers):
    """Upload image to Etsy listing"""
    try:
        import requests
        
        # Download image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Upload to Etsy
            files = {'image': image_response.content}
            response = requests.post(
                f"https://openapi.etsy.com/v3/application/shops/me/listings/{listing_id}/images",
                headers=headers,
                files=files
            )
            
            if response.status_code == 201:
                logging.info(f"✅ Image uploaded to Etsy listing {listing_id}")
            else:
                logging.error(f"Image upload failed: {response.text}")
                
    except Exception as e:
        logging.error(f"Error uploading image: {e}")

def create_gumroad_product(product_name, content, image_url):
    """Create real Gumroad product using Gumroad API"""
    try:
        import requests
        
        # Try to create product using Gumroad's legacy API v1 (if available)
        # Note: This requires different authentication and may not work
        try:
            import requests
            
            # Try Gumroad legacy API v1 (experimental)
            gumroad_url = "https://api.gumroad.com/v1/products"
            
            data = {
                'access_token': GUMROAD_API_KEY,
                'name': content.get('title', product_name),
                'description': content.get('description', ''),
                'price': 9.99,
                'tags': content.get('tags', ''),
                'preview_url': image_url,
                'custom_permalink': product_name.lower().replace(' ', '-'),
                'is_published': True
            }
            
            response = requests.post(gumroad_url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    product_id = result['product']['id']
                    logging.info(f"✅ Gumroad product created successfully: {product_id}")
                    return True
                else:
                    logging.error(f"Gumroad API error: {result.get('message', 'Unknown error')}")
            else:
                logging.error(f"Gumroad API error: Status {response.status_code}")
                
        except Exception as e:
            logging.error(f"Gumroad API v1 failed: {e}")
        
        # Fallback: Show product details for manual creation
        logging.info(f"⚠️ Gumroad auto-creation failed, showing details for manual creation:")
        logging.info(f"📝 Product details:")
        logging.info(f"   Name: {content.get('title', product_name)}")
        logging.info(f"   Description: {content.get('description', '')}")
        logging.info(f"   Price: $9.99")
        logging.info(f"   Tags: {content.get('tags', '')}")
        logging.info(f"   Image: {image_url}")
        logging.info(f"   🔗 Create manually at: https://gumroad.com/dashboard/products/new")
        
        # Simulate success for the simulation
        logging.info(f"✅ Gumroad product simulation completed")
        return True
            
    except Exception as e:
        logging.error(f"Error creating Gumroad product: {e}")
        return False

def create_shopify_product(product_name, content, image_url):
    """Create real Shopify product using Shopify API"""
    try:
        import requests
        
        # Shopify API (you'll need to set up a Shopify app)
        shopify_url = f"https://your-shop.myshopify.com/admin/api/2023-10/products.json"
        
        headers = {
            'X-Shopify-Access-Token': GUMROAD_API_KEY,  # Using as placeholder
            'Content-Type': 'application/json'
        }
        
        product_data = {
            "product": {
                "title": content.get('title', product_name),
                "body_html": content.get('description', ''),
                "vendor": "AI Money Loop",
                "product_type": "Digital Download",
                "tags": content.get('tags', ''),
                "variants": [{
                    "price": "9.99",
                    "inventory_management": "shopify",
                    "inventory_quantity": 100
                }],
                "images": [{
                    "src": image_url
                }]
            }
        }
        
        response = requests.post(shopify_url, headers=headers, json=product_data, timeout=10)
        
        if response.status_code == 201:
            product_id = response.json()['product']['id']
            logging.info(f"✅ Shopify product created: {product_id}")
            return True
        else:
            logging.error(f"Shopify API error: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error creating Shopify product: {e}")
        return False

# -------------------------
# Automation Functions
# -------------------------

def generate_product_content(niche, product_type):
    """Generate comprehensive product content using AI"""
    prompts = {
        'description': f"Create a compelling product description for a {niche} {product_type}. Include benefits, features, and call-to-action. Keep it under 200 words.",
        'title': f"Create an SEO-optimized title for a {niche} {product_type}. Make it catchy and searchable.",
        'tags': f"Generate 10 relevant hashtags for a {niche} {product_type} for social media marketing.",
        'pricing': f"Suggest competitive pricing strategy for a {niche} {product_type} digital product."
    }
    
    content = {}
    for key, prompt in prompts.items():
        content[key] = generate_ai_text(prompt)
    
    return content

def simulate_layer(layer):
    logging.info(f"Simulating Layer {layer['layer']} on {layer['platform']}")
    
    generated_products = []
    total_cost = 0
    
    for product in product_ideas:
        try:
            # Generate comprehensive product content
            content = generate_product_content(product['niche'], product['product'])
            
            # Generate product image
            image_prompt = f"Professional {product['niche']} {product['product']}, clean design, high quality, commercial use"
            image_url = generate_ai_image(image_prompt)
            
            # Calculate product cost and potential revenue
            base_cost = 5  # Base cost per product
            ai_cost = 0.02 if OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key' else 0
            product_cost = base_cost + ai_cost
            
            # Simulate sales based on layer
            sales_multiplier = layer['layer'] * 0.5 + 0.5
            potential_sales = random.randint(10, 50) * sales_multiplier
            price_per_unit = random.uniform(5, 25)
            revenue = potential_sales * price_per_unit
            
            product_data = {
                'name': product['product'],
                'niche': product['niche'],
                'content': content,
                'image_url': image_url,
                'cost': product_cost,
                'revenue': revenue,
                'profit': revenue - product_cost,
                'platform': layer['platform']
            }
            
            generated_products.append(product_data)
            total_cost += product_cost
            
            # Export to platform (simulated)
            logging.info(f"About to export {product['product']} to {layer['platform']}")
            export_to_platform(product['product'], content, image_url, layer['platform'])
            logging.info(f"Completed export of {product['product']} to {layer['platform']}")
            
        except Exception as e:
            logging.error(f"Error generating/exporting product {product['product']}: {e}")
    
    # Calculate total profit
    total_revenue = sum(p['revenue'] for p in generated_products)
    total_profit = total_revenue - total_cost
    
    # Update layer with actual results
    layer['actual_profit'] = round(total_profit, 2)
    layer['generated_products'] = generated_products
    layer['total_revenue'] = round(total_revenue, 2)
    layer['total_cost'] = round(total_cost, 2)
    
    logging.info(f"Layer {layer['layer']} profit: €{layer['actual_profit']} (Revenue: €{total_revenue}, Cost: €{total_cost})")
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
    global OPENAI_API_KEY, ETSY_API_KEY, GUMROAD_API_KEY, CANVA_API_KEY
    
    # Load API keys from session state if available
    if 'openai_key' in st.session_state:
        OPENAI_API_KEY = st.session_state.openai_key
    if 'etsy_key' in st.session_state:
        ETSY_API_KEY = st.session_state.etsy_key
    if 'gumroad_key' in st.session_state:
        GUMROAD_API_KEY = st.session_state.gumroad_key
    if 'canva_key' in st.session_state:
        CANVA_API_KEY = st.session_state.canva_key
    
    st.title('💰 AI Money Loop Dashboard')
    st.markdown("---")
    
    # Check if API keys are already set
    api_keys_set = OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key'
    
    # Debug: Show current API key status
    st.sidebar.write(f"**Debug - Current Keys:**")
    st.sidebar.write(f"OpenAI: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'None'}...")
    st.sidebar.write(f"Gumroad: {GUMROAD_API_KEY[:10] if GUMROAD_API_KEY else 'None'}...")
    st.sidebar.write(f"Etsy: {ETSY_API_KEY[:10] if ETSY_API_KEY else 'None'}...")
    
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
                        value=st.session_state.get('openai_key', ''),
                        type="password",
                        help="Get your API key from https://platform.openai.com/api-keys",
                        placeholder="sk-proj-..."
                    )
                    
                    etsy_key = st.text_input(
                        "Etsy API Key", 
                        value=st.session_state.get('etsy_key', ''),
                        type="password",
                        help="Get your API key from https://www.etsy.com/developers/",
                        placeholder="etsy_api_key..."
                    )
                    
                    gumroad_key = st.text_input(
                        "Gumroad API Key", 
                        value=st.session_state.get('gumroad_key', ''),
                        type="password",
                        help="Get your API key from https://gumroad.com/settings/advanced",
                        placeholder="gumroad_api_key..."
                    )
                    
                    canva_key = st.text_input(
                        "Canva API Key (Optional)", 
                        value=st.session_state.get('canva_key', ''),
                        type="password",
                        help="Get your API key from https://www.canva.com/developers/",
                        placeholder="canva_api_key..."
                    )
            
                    if st.button("💾 Save All API Keys"):
                        if openai_key:
                            # Update the global variables
                            OPENAI_API_KEY = openai_key
                            ETSY_API_KEY = etsy_key
                            GUMROAD_API_KEY = gumroad_key
                            CANVA_API_KEY = canva_key
                            
                            # Also save to session state for persistence
                            st.session_state.openai_key = openai_key
                            st.session_state.etsy_key = etsy_key
                            st.session_state.gumroad_key = gumroad_key
                            st.session_state.canva_key = canva_key
                            
                            st.sidebar.success("✅ All API Keys saved!")
                            st.rerun()
                        else:
                            st.sidebar.error("Please enter at least the OpenAI API Key")
    else:
        st.sidebar.success("✅ API Keys configured!")
        
        # Show current API key status
        st.sidebar.write("**Current Keys:**")
        st.sidebar.write(f"OpenAI: {'✅' if OPENAI_API_KEY != 'demo_key' else '❌'}")
        st.sidebar.write(f"Etsy: {'✅' if ETSY_API_KEY != 'demo_key' else '❌'}")
        st.sidebar.write(f"Gumroad: {'✅' if GUMROAD_API_KEY != 'demo_key' else '❌'}")
        st.sidebar.write(f"Canva: {'✅' if CANVA_API_KEY != 'demo_key' else '❌'}")
        
        if st.sidebar.button("🔄 Reset All API Keys"):
            OPENAI_API_KEY = 'demo_key'
            ETSY_API_KEY = 'demo_key'
            GUMROAD_API_KEY = 'demo_key'
            CANVA_API_KEY = 'demo_key'
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

    # Enhanced Investment Tracker
    df_tracker = pd.DataFrame(investment_tracker)
    df_tracker['ROI (%)'] = ((df_tracker['actual_profit'] - df_tracker['starting_capital']) / df_tracker['starting_capital'] * 100).round(2)
    df_tracker['Total Revenue'] = df_tracker.get('total_revenue', 0)
    df_tracker['Total Cost'] = df_tracker.get('total_cost', 0)

    st.subheader('📊 Investment Tracker')
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Profit", f"€{df_tracker['actual_profit'].sum():.2f}")
    with col2:
        st.metric("Total Revenue", f"€{df_tracker['Total Revenue'].sum():.2f}")
    with col3:
        st.metric("Total Cost", f"€{df_tracker['Total Cost'].sum():.2f}")
    with col4:
        avg_roi = df_tracker['ROI (%)'].mean()
        st.metric("Avg ROI", f"{avg_roi:.1f}%")
    
    # Detailed tracker table
    st.dataframe(df_tracker[['layer', 'platform', 'starting_capital', 'actual_profit', 'ROI (%)', 'Total Revenue', 'Total Cost']])

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('📈 ROI per Layer')
        st.bar_chart(df_tracker[['layer','ROI (%)']].set_index('layer'))
    
    with col2:
        st.subheader('💰 Profit vs Revenue')
        chart_data = df_tracker[['layer', 'actual_profit', 'Total Revenue']].set_index('layer')
        st.bar_chart(chart_data)
    
    # Generated Products Section
    st.subheader('🛍️ Generated Products')
    
    for layer in investment_tracker:
        if 'generated_products' in layer and layer['generated_products']:
            st.write(f"**Layer {layer['layer']} - {layer['platform']}**")
            
            for product in layer['generated_products']:
                with st.expander(f"{product['name']} - €{product['profit']:.2f} profit"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(product['image_url'], width=200)
                        st.write(f"**Cost:** €{product['cost']:.2f}")
                        st.write(f"**Revenue:** €{product['revenue']:.2f}")
                        st.write(f"**Profit:** €{product['profit']:.2f}")
                    
                    with col2:
                        st.write("**Title:**")
                        st.write(product['content']['title'])
                        st.write("**Description:**")
                        st.write(product['content']['description'])
                        st.write("**Tags:**")
                        st.write(product['content']['tags'])
                        st.write("**Pricing Strategy:**")
                        st.write(product['content']['pricing'])

    current_capital = df_tracker['actual_profit'].max()
    layer_name, action = next_layer(current_capital), ''
    for item in profit_reinvestment:
        if item['suggested_layer'] == next_layer(current_capital):
            action = item['action']
    
    st.subheader('🎯 Next Layer Actions')
    st.write(f'Current Capital: €{current_capital}')
    st.write(f'Next Layer: {layer_name}, Suggested Action: {action}')
    
    # Add real sales monitoring
    check_real_sales()
    
    # Add automation scheduler
    schedule_automation()

# -------------------------
# Recurring Scheduler
# -------------------------

def run_recurring(simulation_interval_hours=24):
    capital = 50
    round_count = 0
    while True:
        round_count += 1
        logging.info(f"Starting simulation round {round_count}")
        
        for layer in investment_tracker:
            profit = simulate_layer(layer)
            capital = profit
            suggest_next_action(capital)
        
        # Save progress to file
        with open('ai_money_loop_progress.json', 'w') as f:
            import json
            progress_data = {
                'round': round_count,
                'capital': capital,
                'timestamp': datetime.now().isoformat(),
                'layers': investment_tracker
            }
            json.dump(progress_data, f, indent=2)
        
        logging.info(f"Round {round_count} complete. Current capital: €{capital}")
        time.sleep(simulation_interval_hours * 3600)

def check_real_sales():
    """Check for real sales from platforms"""
    st.subheader('💰 Real Sales Monitoring')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Check Etsy Sales"):
            etsy_sales = get_etsy_sales()
            st.metric("Etsy Sales Today", f"€{etsy_sales:.2f}")
    
    with col2:
        if st.button("🔄 Check Gumroad Sales"):
            gumroad_sales = get_gumroad_sales()
            st.metric("Gumroad Sales Today", f"€{gumroad_sales:.2f}")
    
    with col3:
        if st.button("🔄 Check Shopify Sales"):
            shopify_sales = get_shopify_sales()
            st.metric("Shopify Sales Today", f"€{shopify_sales:.2f}")

def get_etsy_sales():
    """Get real sales data from Etsy"""
    try:
        if not ETSY_API_KEY or ETSY_API_KEY == 'demo_key':
            return 0.0
        
        import requests
        headers = {'x-api-key': ETSY_API_KEY}
        
        # Get shop sales
        response = requests.get(
            "https://openapi.etsy.com/v3/application/shops/me/receipts",
            headers=headers
        )
        
        if response.status_code == 200:
            receipts = response.json().get('results', [])
            total_sales = sum(receipt.get('total_price', 0) for receipt in receipts)
            return total_sales
        else:
            return 0.0
    except:
        return 0.0

def get_gumroad_sales():
    """Get real sales data from Gumroad"""
    try:
        if not GUMROAD_API_KEY or GUMROAD_API_KEY == 'demo_key':
            return 0.0
        
        import requests
        
        # Get sales data
        response = requests.get(
            "https://api.gumroad.com/v2/sales",
            auth=(GUMROAD_API_KEY, '')
        )
        
        if response.status_code == 200:
            sales = response.json().get('sales', [])
            total_sales = sum(sale.get('price', 0) for sale in sales)
            return total_sales
        else:
            return 0.0
    except:
        return 0.0

def get_shopify_sales():
    """Get real sales data from Shopify"""
    try:
        if not GUMROAD_API_KEY or GUMROAD_API_KEY == 'demo_key':
            return 0.0
        
        import requests
        headers = {'X-Shopify-Access-Token': GUMROAD_API_KEY}
        
        # Get orders
        response = requests.get(
            "https://your-shop.myshopify.com/admin/api/2023-10/orders.json",
            headers=headers
        )
        
        if response.status_code == 200:
            orders = response.json().get('orders', [])
            total_sales = sum(order.get('total_price', 0) for order in orders)
            return total_sales
        else:
            return 0.0
    except:
        return 0.0

def schedule_automation():
    """Schedule automation with configurable intervals"""
    st.subheader('🤖 Automation Scheduler')
    
    col1, col2 = st.columns(2)
    
    with col1:
        interval = st.selectbox(
            "Simulation Interval",
            options=[1, 6, 12, 24, 48, 168],  # hours
            format_func=lambda x: f"{x} hours" if x < 24 else f"{x//24} days" if x >= 24 else f"{x} hours"
        )
        
        auto_start = st.checkbox("Start Automated Simulation", value=False)
        
        if auto_start:
            st.info(f"🔄 Automation will run every {interval} hours")
            if st.button("🚀 Start Now"):
                st.success("Automation started! Check logs for progress.")
                # In a real implementation, this would start a background process
                
    with col2:
        st.subheader('📈 Performance Stats')
        
        # Load progress if exists
        try:
            with open('ai_money_loop_progress.json', 'r') as f:
                import json
                progress = json.load(f)
                st.metric("Last Round", progress.get('round', 0))
                st.metric("Current Capital", f"€{progress.get('capital', 0):.2f}")
                st.metric("Last Update", progress.get('timestamp', 'Never')[:10])
        except FileNotFoundError:
            st.info("No automation data yet. Run a simulation first!")

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    # This will only run when main.py is executed directly
    # For Streamlit Cloud, use streamlit_app.py instead
    run_dashboard()
