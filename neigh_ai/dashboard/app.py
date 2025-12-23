import streamlit as st

st.set_page_config(
    page_title="Starter Dashboard",
    layout="wide",
)

st.title("📊 Starter Streamlit + Plotly Dashboard")
st.write("Use the sidebar to navigate between pages.")

st.markdown("""
This is the **main landing page**.

Pages included:
- Overview (with sidebar)
- Analytics (Plotly charts)
- About

Run with:
```bash
streamlit run app.py
```
""")
