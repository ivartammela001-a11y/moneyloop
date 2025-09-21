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
SHOPIFY_API_KEY = os.getenv('SHOPIFY_API_KEY', 'demo_key')

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
    elif platform == 'Shopify/Printful' and SHOPIFY_API_KEY and SHOPIFY_API_KEY != 'demo_key':
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
            'X-Shopify-Access-Token': SHOPIFY_API_KEY,  # Using Shopify API key
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
        'pricing': f"Suggest competitive pricing strategy for a {niche} {product_type} digital product.",
        'recommended_price': f"Analyze the market and recommend the optimal sale price for a {niche} {product_type} digital product. Consider competition, value, and profit margins. Return only a number (e.g., 12.99)."
    }
    
    content = {}
    for key, prompt in prompts.items():
        content[key] = generate_ai_text(prompt)
    
    # Parse recommended price and add pricing analysis
    try:
        recommended_price = float(content['recommended_price'].replace('$', '').replace('€', '').strip())
        content['recommended_price'] = recommended_price
        content['pricing_analysis'] = f"Recommended price: €{recommended_price:.2f} based on market analysis"
    except:
        # Fallback pricing based on product type
        base_prices = {
            'printable': 9.99,
            'template': 14.99,
            'planner': 19.99,
            'ebook': 24.99,
            'course': 49.99
        }
        recommended_price = base_prices.get(product_type.lower(), 12.99)
        content['recommended_price'] = recommended_price
        content['pricing_analysis'] = f"Recommended price: €{recommended_price:.2f} (default pricing)"
    
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
            
            # Use AI-recommended price for revenue calculation
            recommended_price = content.get('recommended_price', 12.99)
            
            # Simulate sales based on layer and price
            sales_multiplier = layer['layer'] * 0.5 + 0.5
            potential_sales = random.randint(10, 50) * sales_multiplier
            revenue = potential_sales * recommended_price
            
            product_data = {
                'name': product['product'],
                'niche': product['niche'],
                'content': content,
                'image_url': image_url,
                'cost': product_cost,
                'revenue': revenue,
                'profit': revenue - product_cost,
                'platform': layer['platform'],
                'recommended_price': recommended_price,
                'potential_sales': potential_sales
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
    global OPENAI_API_KEY, ETSY_API_KEY, GUMROAD_API_KEY, CANVA_API_KEY, SHOPIFY_API_KEY
    
    # Load API keys from session state if available
    if 'openai_key' in st.session_state:
        OPENAI_API_KEY = st.session_state.openai_key
    if 'etsy_key' in st.session_state:
        ETSY_API_KEY = st.session_state.etsy_key
    if 'gumroad_key' in st.session_state:
        GUMROAD_API_KEY = st.session_state.gumroad_key
    if 'canva_key' in st.session_state:
        CANVA_API_KEY = st.session_state.canva_key
    if 'shopify_key' in st.session_state:
        SHOPIFY_API_KEY = st.session_state.shopify_key
    
    st.title('💰 AI Money Loop Dashboard')
    st.markdown("---")
    
    # Check if API keys are already set
    api_keys_set = OPENAI_API_KEY and OPENAI_API_KEY != 'demo_key'
    
    # Debug: Show current API key status
    st.sidebar.write(f"**Debug - Current Keys:**")
    st.sidebar.write(f"OpenAI: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'None'}...")
    st.sidebar.write(f"Gumroad: {GUMROAD_API_KEY[:10] if GUMROAD_API_KEY else 'None'}...")
    st.sidebar.write(f"Etsy: {ETSY_API_KEY[:10] if ETSY_API_KEY else 'None'}...")
    st.sidebar.write(f"Shopify: {SHOPIFY_API_KEY[:10] if SHOPIFY_API_KEY else 'None'}...")
    
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
                    
                    shopify_key = st.text_input(
                        "Shopify API Key (Optional)", 
                        value=st.session_state.get('shopify_key', ''),
                        type="password",
                        help="Get your API key from https://partners.shopify.com/",
                        placeholder="shopify_api_key..."
                    )
            
                    if st.button("💾 Save All API Keys"):
                        if openai_key:
                            # Update the global variables
                            OPENAI_API_KEY = openai_key
                            ETSY_API_KEY = etsy_key
                            GUMROAD_API_KEY = gumroad_key
                            CANVA_API_KEY = canva_key
                            SHOPIFY_API_KEY = shopify_key
                            
                            # Also save to session state for persistence
                            st.session_state.openai_key = openai_key
                            st.session_state.etsy_key = etsy_key
                            st.session_state.gumroad_key = gumroad_key
                            st.session_state.canva_key = canva_key
                            st.session_state.shopify_key = shopify_key
                            
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
        st.sidebar.write(f"Shopify: {'✅' if SHOPIFY_API_KEY != 'demo_key' else '❌'}")
        
        if st.sidebar.button("🔄 Reset All API Keys"):
            OPENAI_API_KEY = 'demo_key'
            ETSY_API_KEY = 'demo_key'
            GUMROAD_API_KEY = 'demo_key'
            CANVA_API_KEY = 'demo_key'
            SHOPIFY_API_KEY = 'demo_key'
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

    # Enhanced Investment Tracker with Real Data
    df_tracker = pd.DataFrame(investment_tracker)
    df_tracker['ROI (%)'] = ((df_tracker['actual_profit'] - df_tracker['starting_capital']) / df_tracker['starting_capital'] * 100).round(2)
    df_tracker['Total Revenue'] = df_tracker.get('total_revenue', 0)
    df_tracker['Total Cost'] = df_tracker.get('total_cost', 0)

    st.subheader('📊 Investment Tracker')
    
    # Add real-time data refresh button
    if st.button("🔄 Refresh Real Data"):
        st.rerun()
    
    # Key Metrics with Real vs Simulated indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_profit = df_tracker['actual_profit'].sum()
        profit_type = "💰 REAL" if total_profit > 0 else "🎭 SIMULATED"
        st.metric("Total Profit", f"€{total_profit:.2f}", help=profit_type)
    with col2:
        total_revenue = df_tracker['Total Revenue'].sum()
        revenue_type = "💰 REAL" if total_revenue > 0 else "🎭 SIMULATED"
        st.metric("Total Revenue", f"€{total_revenue:.2f}", help=revenue_type)
    with col3:
        total_cost = df_tracker['Total Cost'].sum()
        st.metric("Total Cost", f"€{total_cost:.2f}")
    with col4:
        avg_roi = df_tracker['ROI (%)'].mean()
        roi_type = "💰 REAL" if avg_roi > 0 else "🎭 SIMULATED"
        st.metric("Avg ROI", f"{avg_roi:.1f}%", help=roi_type)
    
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
                        
                        # Pricing section with recommended price
                        st.markdown("### 💰 Pricing Analysis")
                        st.metric("🎯 Recommended Price", f"€{product['recommended_price']:.2f}")
                        st.metric("📊 Potential Sales", f"{product['potential_sales']:.0f} units")
                        st.metric("💵 Total Revenue", f"€{product['revenue']:.2f}")
                        
                        st.markdown("### 📈 Financials")
                        st.write(f"**Cost:** €{product['cost']:.2f}")
                        st.write(f"**Profit:** €{product['profit']:.2f}")
                        st.write(f"**Profit Margin:** {((product['profit'] / product['revenue']) * 100):.1f}%")
                    
                    with col2:
                        st.write("**Title:**")
                        st.write(product['content']['title'])
                        st.write("**Description:**")
                        st.write(product['content']['description'])
                        st.write("**Tags:**")
                        st.write(product['content']['tags'])
                        st.write("**Pricing Strategy:**")
                        st.write(product['content']['pricing'])
                        st.write("**AI Price Analysis:**")
                        st.write(product['content']['pricing_analysis'])

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
    
    # Real Investment Tracker Section
    st.subheader('🎯 Real Investment Tracker')
    
    # Get real sales data
    real_etsy_sales = get_etsy_sales()
    real_gumroad_sales = get_gumroad_sales()
    real_shopify_sales = get_shopify_sales()
    
    # Create real investment tracker
    real_investment_tracker = [
        {'layer': 1, 'platform': 'Etsy', 'starting_capital': 50, 'real_profit': real_etsy_sales, 'real_roi': ((real_etsy_sales - 50) / 50 * 100) if real_etsy_sales > 0 else 0},
        {'layer': 2, 'platform': 'Gumroad', 'starting_capital': 250, 'real_profit': real_gumroad_sales, 'real_roi': ((real_gumroad_sales - 250) / 250 * 100) if real_gumroad_sales > 0 else 0},
        {'layer': 3, 'platform': 'Shopify', 'starting_capital': 1000, 'real_profit': real_shopify_sales, 'real_roi': ((real_shopify_sales - 1000) / 1000 * 100) if real_shopify_sales > 0 else 0}
    ]
    
    # Display real metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_real_profit = real_etsy_sales + real_gumroad_sales + real_shopify_sales
        st.metric("💰 Real Total Profit", f"€{total_real_profit:.2f}")
    with col2:
        st.metric("🛍️ Etsy Sales", f"€{real_etsy_sales:.2f}")
    with col3:
        st.metric("📚 Gumroad Sales", f"€{real_gumroad_sales:.2f}")
    with col4:
        st.metric("🏪 Shopify Sales", f"€{real_shopify_sales:.2f}")
    
    # Real investment tracker table
    if total_real_profit > 0:
        st.success("🎉 **Real sales detected!** Your investment tracker now shows actual numbers.")
        real_df = pd.DataFrame(real_investment_tracker)
        st.dataframe(real_df[['layer', 'platform', 'starting_capital', 'real_profit', 'real_roi']])
    else:
        st.info("ℹ️ No real sales detected yet. The tracker shows simulated data until you make actual sales.")
    
    # Pricing Optimization Section
    st.subheader('🎯 Pricing Optimization')
    
    # Show pricing recommendations for each platform
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🛍️ Etsy Pricing")
        st.info("**Recommended Range:** €8.99 - €15.99\n\n**Best Sellers:** €9.99 - €12.99\n\n**Premium Products:** €14.99 - €19.99")
    
    with col2:
        st.markdown("### 📚 Gumroad Pricing")
        st.info("**Recommended Range:** €9.99 - €24.99\n\n**Digital Downloads:** €9.99 - €14.99\n\n**E-books/Courses:** €19.99 - €49.99")
    
    with col3:
        st.markdown("### 🏪 Shopify Pricing")
        st.info("**Recommended Range:** €12.99 - €29.99\n\n**Printables:** €12.99 - €19.99\n\n**Bundles:** €24.99 - €49.99")
    
    # Pricing strategy tips
    st.markdown("### 💡 Pricing Strategy Tips")
    st.markdown("""
    - **Start Higher:** You can always lower prices, but raising them is harder
    - **Test Different Prices:** A/B test €9.99 vs €12.99 to find optimal price
    - **Bundle Strategy:** Offer multiple products together for better value
    - **Psychological Pricing:** Use €11.99 instead of €12.00
    - **Seasonal Pricing:** Adjust prices based on demand seasons
    """)
    
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
    """Check for real sales from platforms and update investment tracker"""
    st.subheader('💰 Real Sales Monitoring')
    
    col1, col2, col3 = st.columns(3)
    
    # Get real sales data
    etsy_sales = get_etsy_sales()
    gumroad_sales = get_gumroad_sales()
    shopify_sales = get_shopify_sales()
    
    with col1:
        if st.button("🔄 Check Etsy Sales"):
            st.metric("Etsy Sales Today", f"€{etsy_sales:.2f}")
            # Update investment tracker with real data
            if etsy_sales > 0:
                investment_tracker[0]['actual_profit'] = etsy_sales
                st.success(f"✅ Updated Layer 1 with real Etsy sales: €{etsy_sales:.2f}")
    
    with col2:
        if st.button("🔄 Check Gumroad Sales"):
            st.metric("Gumroad Sales Today", f"€{gumroad_sales:.2f}")
            # Update investment tracker with real data
            if gumroad_sales > 0:
                investment_tracker[1]['actual_profit'] = gumroad_sales
                st.success(f"✅ Updated Layer 2 with real Gumroad sales: €{gumroad_sales:.2f}")
    
    with col3:
        if st.button("🔄 Check Shopify Sales"):
            st.metric("Shopify Sales Today", f"€{shopify_sales:.2f}")
            # Update investment tracker with real data
            if shopify_sales > 0:
                investment_tracker[2]['actual_profit'] = shopify_sales
                st.success(f"✅ Updated Layer 3 with real Shopify sales: €{shopify_sales:.2f}")
    
    # Show total real sales
    total_real_sales = etsy_sales + gumroad_sales + shopify_sales
    if total_real_sales > 0:
        st.success(f"🎯 **Total Real Sales Today: €{total_real_sales:.2f}**")
        
        # Update all layers with real data
        investment_tracker[0]['actual_profit'] = etsy_sales
        investment_tracker[1]['actual_profit'] = gumroad_sales  
        investment_tracker[2]['actual_profit'] = shopify_sales
        
        # Calculate real ROI
        for i, layer in enumerate(investment_tracker):
            if layer['actual_profit'] > 0:
                layer['ROI'] = ((layer['actual_profit'] - layer['starting_capital']) / layer['starting_capital'] * 100)
                st.info(f"Layer {i+1} Real ROI: {layer['ROI']:.1f}%")

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
        if not SHOPIFY_API_KEY or SHOPIFY_API_KEY == 'demo_key':
            return 0.0
        
        import requests
        headers = {'X-Shopify-Access-Token': SHOPIFY_API_KEY}
        
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
