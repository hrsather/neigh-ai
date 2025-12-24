import streamlit as st

st.set_page_config(
    page_title="Happy Birthday Josh 🎉",
    layout="wide",
)

st.title("🎂 Happy Birthday, Josh! 🐎")

st.markdown(
    """
    Welcome to your **Horse Racing Analytics Dashboard**.

    This site was built to explore, visualize, and analyze horse racing data — from
    individual performances to family-lineage trends that influence race outcomes.
    """
)

st.divider()

st.subheader("What you can do here")

st.markdown(
    """
    🏇 **Explore horses**
    - Browse horses and view detailed race results
    - See performance trends across distances and speeds

    📊 **Analyze rankings**
    - Compare horses by performance score
    - Identify top performers by percentile and consistency

    🌳 **Understand lineage impact**
    - Visualize how sires, dams, siblings, and cousins influence outcomes
    - See how family history predicts future performance

    📈 **Interactive visuals**
    - Filter, sort, and search large datasets
    - Click into horses for detailed plots and comparisons
    """
)

st.divider()

st.markdown(
    """
    Use the **sidebar** to navigate between pages and start exploring.
    Whether you're digging into raw results or high-level rankings, this dashboard
    is designed to make horse racing data intuitive, fast, and fun.

    Enjoy the day — and enjoy the data! 🎉
    """
)
