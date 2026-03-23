import streamlit as st
import requests
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000/analyze"

st.title("Journal Recommendation Engine")

abstract = st.text_area("Paste your abstract")

journals = st.multiselect(
    "Select candidate journals",
    [
        "Computer Networks",
        "IEEE Transactions on AI",
        "Bioinformatics"
    ]
)

if st.button("Analyze") and abstract and journals:
    response = requests.post(API_URL, json={
        "abstract": abstract,
        "candidate_journals": journals
    })

    data = response.json()

    st.subheader("Final Recommendation")
    st.success(data["final_recommendation"])

    # Confidence bar chart
    names = [j["journal"] for j in data["ranked_journals"]]
    scores = [j["confidence"] for j in data["ranked_journals"]]

    fig, ax = plt.subplots()
    ax.bar(names, scores)
    ax.axhline(0.65, linestyle="--")
    ax.axhline(0.45, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")

    st.pyplot(fig)