import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(dotenv_path=env_path, override=True)
# client = OpenAI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# --- Data loading -----------------------------------------------------------
@st.cache_data
def load_data():
    """Return Gapminder data for the year 2007 only."""
    return px.data.gapminder().query("year == 2007")

df = load_data()

# --- Sidebar ----------------------------------------------------------------
st.sidebar.title("Dashboard Controls")
continent = st.sidebar.selectbox("Select Continent", df["continent"].unique())
filtered_df = df[df["continent"] == continent]

# --- Page title -------------------------------------------------------------
st.title("📈 Interactive Dashboard: Gapminder 2007")

# --- Visualization 1 – Scatter --------------------------------------------
st.subheader("GDP per Capita vs Life Expectancy")
fig1 = px.scatter(
    filtered_df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="country",
    log_x=True,
    size_max=60,
    title=f"GDP vs Life Expectancy in {continent}",
)
st.plotly_chart(fig1)

# --- Visualization 2 – Bar --------------------------------------------------
st.subheader("Top 10 Countries by GDP per Capita")

top10_gdp = filtered_df.nlargest(10, "gdpPercap")
fig2, ax = plt.subplots()
ax.barh(top10_gdp["country"], top10_gdp["gdpPercap"])
ax.set_xlabel("GDP per Capita")
ax.invert_yaxis()
st.pyplot(fig2)

# --- Visualization 3 – Histogram -------------------------------------------
st.subheader("Distribution of Life Expectancy")
fig3 = px.histogram(
    filtered_df,
    x="lifeExp",
    nbins=10,
    title="Life Expectancy Distribution",
)
st.plotly_chart(fig3)

# --- AI‑Powered summary -----------------------------------------------------
st.subheader("🧠 AI‑Powered Summary")

if st.button("Generate Summary"):
    prompt = (
        "Provide a brief, insightful summary of the following data:\n" +
        filtered_df.describe(include="all").to_string()
    )

    with st.spinner("Generating summary…"):
        try:
            completion = client.chat.completions.create(
                model="o4-mini-2025-04-16",  # use the latest model available to your account
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": prompt},
                ],
            )
            summary = completion.choices[0].message.content
            st.success("Summary Generated:")
            st.write(summary)
        except Exception as e:
            st.error(f"❌ Failed to generate summary: {e}")

# --- Footer -----------------------------------------------------------------
st.markdown("---")
st.markdown("Data Source: [Gapminder](https://www.gapminder.org/data/)")
