import pandas as pd
from src.phase2.abstract_pipeline import run_phase2


def test_novel_abstract_no_recommendation():
    df = pd.read_csv("data/master_journals_enriched.csv")

    result = run_phase2(
        user_abstract="""
        We introduce a bio-inspired quantum swarm optimization
        framework for neural-symbolic reasoning in molecular robotics.
        """,
        candidate_journals=[{"journal_name": "Computer Networks"}],
        df=df
    )

    assert result is not None
    assert "recommended_headings" in result
    assert result["recommended_headings"] is None
    assert "novel" in result["final_decision"].lower()


def test_strong_match_recommends_journal():
    df = pd.read_csv("data/master_journals_enriched.csv")

    result = run_phase2(
        user_abstract="""
        This paper studies graph spanner constructions for optimizing
        routing efficiency in large-scale communication networks.
        Latency and bandwidth trade-offs are evaluated.
        """,
        candidate_journals=[{"journal_name": "Computer Networks"}],
        df=df
    )

    assert result is not None
    assert result["final_decision"].lower().startswith("strong")
    assert result["recommended_headings"] is not None
    assert len(result["recommended_headings"]) >= 1


def test_low_confidence_graceful_fallback():
    df = pd.read_csv("data/master_journals_enriched.csv")

    result = run_phase2(
        user_abstract="A brief note on algorithms.",
        candidate_journals=[{"journal_name": "Computer Networks"}],
        df=df
    )

    assert result is not None
    assert "final_decision" in result
    assert "recommended_headings" in result
