# AI Money Loop System

[![CI](https://github.com/ivartammela001-a11y/moneyloop/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ivartammela001-a11y/moneyloop/actions/workflows/tests.yml?query=branch%3Amain)

A complete AI-powered digital product creation and sales automation system with 3-layer investment strategy.

## 🚀 Quick Start

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run main.py
   ```

### Free Cloud Deployment

#### Option 1: Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

#### Option 2: Railway
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Deploy automatically

#### Option 3: Render
1. Go to [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Deploy for free

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and add your API keys:
- `OPENAI_API_KEY` - Required for AI content generation
- `CANVA_API_KEY` - Optional for design automation
- `ETSY_API_KEY` - Optional for Etsy integration
- `GUMROAD_API_KEY` - Optional for Gumroad integration

### Demo Mode
The app works in demo mode without API keys for testing and simulation.

## 📊 Features

- **3-Layer Investment Strategy**: €50 → €250 → €1000 → €3000
- **AI Content Generation**: Automated product descriptions and images
- **Multi-Platform Support**: Etsy, Gumroad, Shopify/Printful
- **Real-time Analytics**: ROI tracking and profit visualization
- **Smart Recommendations**: Next action suggestions based on capital

## 🎯 Investment Layers

1. **Layer 1 (€50 → €250)**: Etsy digital printables
2. **Layer 2 (€250 → €1000)**: Gumroad templates & e-books
3. **Layer 3 (€1000 → €3000)**: Shopify/Printful POD store

## 📝 Logs
All logs are written to `ai_money_loop.log`.
