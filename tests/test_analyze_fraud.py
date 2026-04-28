"""
Integration tests for score_transactions() and summarize_results().

These tests use the real CSV data files to validate that the pipeline
produces correct metrics end-to-end, including the critical regression
guard that confirmed chargebacks never land in the low-risk bucket.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
DATA = Path(__file__).resolve().parents[1] / "data"

sys.path.insert(0, str(SRC))

from analyze_fraud import score_transactions, summarize_results


# ---------------------------------------------------------------------------
# Fixtures — real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_data():
    accounts = pd.read_csv(DATA / "accounts.csv")
    transactions = pd.read_csv(DATA / "transactions.csv")
    chargebacks = pd.read_csv(DATA / "chargebacks.csv")
    return accounts, transactions, chargebacks


@pytest.fixture(scope="module")
def scored(real_data):
    accounts, transactions, _ = real_data
    return score_transactions(transactions, accounts)


@pytest.fixture(scope="module")
def summary(scored, real_data):
    _, _, chargebacks = real_data
    return summarize_results(scored, chargebacks)


# ---------------------------------------------------------------------------
# score_transactions — output shape and columns
# ---------------------------------------------------------------------------

def test_scored_has_risk_score_column(scored):
    assert "risk_score" in scored.columns


def test_scored_has_risk_label_column(scored):
    assert "risk_label" in scored.columns


def test_scored_row_count_matches_transactions(scored, real_data):
    _, transactions, _ = real_data
    assert len(scored) == len(transactions)


def test_risk_scores_in_valid_range(scored):
    assert scored["risk_score"].between(0, 100).all()


def test_risk_labels_are_valid(scored):
    assert set(scored["risk_label"]).issubset({"low", "medium", "high"})


def test_risk_label_matches_score(scored):
    """risk_label must be consistent with risk_score for every row."""
    for _, row in scored.iterrows():
        score = row["risk_score"]
        label = row["risk_label"]
        if score >= 60:
            assert label == "high", f"score {score} should be high, got {label}"
        elif score >= 30:
            assert label == "medium", f"score {score} should be medium, got {label}"
        else:
            assert label == "low", f"score {score} should be low, got {label}"


# ---------------------------------------------------------------------------
# summarize_results — output shape and columns
# ---------------------------------------------------------------------------

def test_summary_has_required_columns(summary):
    required = {"risk_label", "transactions", "total_amount_usd",
                "avg_amount_usd", "chargebacks", "chargeback_rate"}
    assert required.issubset(set(summary.columns))


def test_summary_transaction_count_matches_total(summary, real_data):
    _, transactions, _ = real_data
    assert summary["transactions"].sum() == len(transactions)


def test_chargeback_rate_is_chargebacks_over_transactions(summary):
    for _, row in summary.iterrows():
        expected = row["chargebacks"] / row["transactions"]
        assert abs(row["chargeback_rate"] - expected) < 1e-9


def test_chargeback_rate_between_0_and_1(summary):
    assert summary["chargeback_rate"].between(0, 1).all()


def test_total_amount_equals_avg_times_count(summary):
    for _, row in summary.iterrows():
        expected = row["avg_amount_usd"] * row["transactions"]
        assert abs(row["total_amount_usd"] - expected) < 0.01


# ---------------------------------------------------------------------------
# Metric quality — regression guards against inverted scoring
# ---------------------------------------------------------------------------

CONFIRMED_CHARGEBACK_IDS = {50003, 50006, 50008, 50011, 50013, 50014, 50015, 50019}


def test_no_confirmed_fraud_in_low_bucket(scored):
    """The critical regression guard: real chargebacks must not score low."""
    low_ids = set(scored[scored["risk_label"] == "low"]["transaction_id"])
    leaked = low_ids & CONFIRMED_CHARGEBACK_IDS
    assert leaked == set(), (
        f"Confirmed fraud transactions scored 'low': {leaked}. "
        "This indicates an inverted signal in risk_rules.py."
    )


def test_high_bucket_chargeback_rate_exceeds_low(summary):
    """High-risk bucket must have a higher fraud rate than low-risk."""
    rates = summary.set_index("risk_label")["chargeback_rate"]
    if "high" in rates and "low" in rates:
        assert rates["high"] > rates["low"], (
            f"high chargeback rate ({rates['high']:.2f}) should exceed "
            f"low chargeback rate ({rates['low']:.2f})"
        )


def test_majority_of_chargebacks_in_high_or_medium(scored, real_data):
    _, _, chargebacks = real_data
    chargeback_ids = set(chargebacks["transaction_id"])
    flagged = scored[scored["transaction_id"].isin(chargeback_ids)]
    not_low = (flagged["risk_label"] != "low").sum()
    total = len(flagged)
    assert not_low / total >= 0.75, (
        f"Only {not_low}/{total} confirmed chargebacks are in medium/high. "
        "Scoring model is not surfacing known fraud."
    )


def test_total_chargeback_count_preserved(summary, real_data):
    _, _, chargebacks = real_data
    assert summary["chargebacks"].sum() == len(chargebacks)


# ---------------------------------------------------------------------------
# Synthetic pipeline tests — summarize_results with controlled inputs
# ---------------------------------------------------------------------------

def make_scored(rows):
    return pd.DataFrame(rows)


def make_chargebacks(ids):
    return pd.DataFrame({"transaction_id": ids})


def test_summarize_all_fraud_in_high():
    scored = make_scored([
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 500},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 500},
    ])
    chargebacks = make_chargebacks([1, 2])
    summary = summarize_results(scored, chargebacks)
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == 1.0
    assert row["chargebacks"] == 2


def test_summarize_no_fraud():
    scored = make_scored([
        {"transaction_id": 1, "risk_label": "low", "amount_usd": 50},
        {"transaction_id": 2, "risk_label": "low", "amount_usd": 75},
    ])
    chargebacks = make_chargebacks([])
    summary = summarize_results(scored, chargebacks)
    row = summary[summary["risk_label"] == "low"].iloc[0]
    assert row["chargeback_rate"] == 0.0
    assert row["chargebacks"] == 0


def test_summarize_partial_fraud():
    scored = make_scored([
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 3, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 4, "risk_label": "high", "amount_usd": 200},
    ])
    chargebacks = make_chargebacks([1, 2])
    summary = summarize_results(scored, chargebacks)
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == 0.5
    assert row["total_amount_usd"] == 800
    assert row["avg_amount_usd"] == 200
