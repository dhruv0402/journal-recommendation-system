import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Journal Recommendation Engine", layout="centered")
st.title("Journal Recommendation Engine")

# ---------------- PHASE 1: TITLE CHECK ----------------
st.header("Step 1: Enter Article Title")

if "title_ok" not in st.session_state:
    st.session_state.title_ok = False

title = st.text_input(
    "Proposed Article Title", placeholder="Enter your article title here..."
)

if st.button("Check Title"):
    if not title.strip():
        st.error("Title cannot be empty")
    else:
        with st.spinner("Checking title..."):
            resp = requests.post(f"{API_BASE}/check-title", json={"title": title})

        if resp.status_code != 200:
            st.error("Title check API error")
        else:
            res = resp.json()
            status = res.get("status")

            if status == "EXACT_MATCH":
                st.error(
                    f"❌ Title already exists in dataset "
                    f"(confidence: {round(res.get('confidence', 0) * 100, 1)}%)."
                )
                st.session_state.title_ok = False

            elif status == "NEAR_MATCH":
                st.warning(
                    f"⚠️ Similar title exists "
                    f"(confidence: {round(res.get('confidence', 0) * 100, 1)}%). "
                    "Consider revising."
                )
                st.session_state.title_ok = False

            else:
                st.success("✅ Title is acceptable. You may proceed.")
                st.session_state.title_ok = True

# ---------------- PHASE 2: ABSTRACT ANALYSIS ----------------
if st.session_state.title_ok:
    st.divider()
    st.header("Step 2: Paste Abstract")

    abstract = st.text_area(
        "Article Abstract",
        height=200,
        placeholder="Paste your research abstract here...",
    )
    if st.button("Analyze Abstract"):
        if not abstract.strip():
            st.error("Abstract cannot be empty")
            st.stop()

        with st.spinner("Analyzing abstract..."):
            response = requests.post(
                f"{API_BASE}/analyze", json={"title": title, "abstract": abstract}
            )

        if response.status_code != 200:
            st.error("Analysis API error")
            st.stop()

        result = response.json()
        # ---------- RAG EXPLANATION ----------
        rag = result.get("rag_explanations")
        if rag:
            st.subheader("AI Explanation")
            if isinstance(rag, dict):
                global_exp = rag.get("global_explanation", {})
                if isinstance(global_exp, dict):
                    st.write(
                        f"**Best Journal (AI):** {global_exp.get('best_journal', 'N/A')}"
                    )
                    st.write(f"**Reason:** {global_exp.get('reason', 'N/A')}")
                    if global_exp.get("best_journal"):
                        st.info(
                            f"Model suggests best fit: {global_exp.get('best_journal')}"
                        )

        # ---------- DUPLICATION HANDLING ----------
        if result.get("status") == "EXACT_MATCH":
            st.error(
                f"❌ Abstract already exists "
                f"(duplication confidence: {round(result.get('duplication_confidence', 0) * 100, 1)}%)."
            )
            st.stop()

        if result.get("status") == "NEAR_DUPLICATE":
            st.warning(
                f"⚠️ Abstract is highly similar to existing work "
                f"(duplication confidence: {round(result.get('duplication_confidence', 0) * 100, 1)}%)."
            )

        # ---------- OUTPUT ----------
        st.success(result.get("final_recommendation", "Analysis complete"))

        ranked = result.get("top3_recommendations", [])

        if ranked:
            st.subheader("Journal Confidence Scores")
            journals = [j.get("journal_name", "Unknown") for j in ranked]
            confidences = [j.get("confidence", 0.0) for j in ranked]
            st.bar_chart(dict(zip(journals, confidences)))

            # ---------- CONFIDENCE TABLE ----------
            st.subheader("Detailed Scores Table")
            df = pd.DataFrame(ranked)
            display_cols = ["journal_name", "confidence", "similarity"]
            st.dataframe(df[display_cols])

            # ---------- COLOR SIGNAL ----------
            st.subheader("Confidence Signal")

            top_conf = confidences[0] if confidences else 0.0

            if top_conf >= 0.7:
                st.success("🟢 Strong Match")
            elif top_conf >= 0.5:
                st.warning("🟡 Moderate Match")
            else:
                st.error("🔴 Weak Match")

            st.subheader("Top Journal Recommendations")

            for j in ranked:
                st.markdown(f"### {j.get('journal_name', 'Unknown')}")
                st.write(f"Confidence: {round(j.get('confidence', 0.0), 3)}")
                st.write(f"Similarity: {round(j.get('similarity', 0.0), 3)}")

                explanation = j.get("explanation", {})
                if explanation:
                    st.write(f"Reason: {explanation.get('reason', 'N/A')}")
                st.markdown("---")

            # ---------- DOWNLOAD REPORT ----------
            st.subheader("Export Report")

            report_text = f"Title: {title}\n\nAbstract:\n{abstract}\n\nTop Journals:\n"
            for j in ranked:
                report_text += f"- {j.get('journal_name')} | Confidence: {j.get('confidence')} | Similarity: {j.get('similarity')}\n"

            st.download_button(
                label="Download Report",
                data=report_text,
                file_name="journal_recommendation.txt",
                mime="text/plain",
            )

        with st.expander("Detailed Results"):
            st.json(result)
