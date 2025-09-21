# Streamlit Cloud entry point
import streamlit as st

# Import and run the main dashboard
try:
    import main
    main.run_dashboard()
except Exception as e:
    st.error(f"Error loading the app: {str(e)}")
    st.write("Please check the logs for more details.")
