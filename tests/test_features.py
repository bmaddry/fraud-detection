import pandas as pd
import pytest
from features import build_model_frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_transactions(rows):
    return pd.DataFrame(rows)


def make_accounts(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def single_transaction():
    return make_transactions([{
        "transaction_id": 1,
        "account_id": 101,
        "amount_usd": 500.0,
        "failed_logins_24h": 0,
    }])


@pytest.fixture
def single_account():
    return make_accounts([{
        "account_id": 101,
        "prior_chargebacks": 0,
        "is_vip": "N",
    }])


# ---------------------------------------------------------------------------
# Merge behaviour
# ---------------------------------------------------------------------------

def test_merge_brings_in_account_fields(single_transaction, single_account):
    df = build_model_frame(single_transaction, single_account)
    assert "prior_chargebacks" in df.columns


def test_merge_preserves_transaction_count(single_transaction, single_account):
    df = build_model_frame(single_transaction, single_account)
    assert len(df) == 1


def test_unknown_account_produces_null_not_dropped():
    """A transaction with no matching account should survive the left join."""
    txns = make_transactions([{"transaction_id": 1, "account_id": 999, "amount_usd": 100, "failed_logins_24h": 0}])
    accounts = make_accounts([{"account_id": 101, "prior_chargebacks": 0}])
    df = build_model_frame(txns, accounts)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["prior_chargebacks"])


def test_multiple_transactions_per_account():
    txns = make_transactions([
        {"transaction_id": 1, "account_id": 101, "amount_usd": 100, "failed_logins_24h": 0},
        {"transaction_id": 2, "account_id": 101, "amount_usd": 200, "failed_logins_24h": 1},
    ])
    accounts = make_accounts([{"account_id": 101, "prior_chargebacks": 2}])
    df = build_model_frame(txns, accounts)
    assert len(df) == 2
    assert (df["prior_chargebacks"] == 2).all()


# ---------------------------------------------------------------------------
# is_large_amount column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected", [
    (0,      0),
    (999.99, 0),
    (1000.0, 1),
    (5000.0, 1),
])
def test_is_large_amount(amount, expected, single_account):
    txns = make_transactions([{"transaction_id": 1, "account_id": 101,
                               "amount_usd": amount, "failed_logins_24h": 0}])
    df = build_model_frame(txns, single_account)
    assert df.iloc[0]["is_large_amount"] == expected


def test_is_large_amount_is_integer(single_transaction, single_account):
    df = build_model_frame(single_transaction, single_account)
    assert df["is_large_amount"].dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# login_pressure column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failed_logins,expected_pressure", [
    (0,  "none"),
    (1,  "low"),
    (2,  "low"),
    (3,  "high"),
    (10, "high"),
])
def test_login_pressure_buckets(failed_logins, expected_pressure, single_account):
    txns = make_transactions([{"transaction_id": 1, "account_id": 101,
                               "amount_usd": 0, "failed_logins_24h": failed_logins}])
    df = build_model_frame(txns, single_account)
    assert str(df.iloc[0]["login_pressure"]) == expected_pressure


def test_login_pressure_only_has_three_categories(single_account):
    txns = make_transactions([
        {"transaction_id": i, "account_id": 101, "amount_usd": 0, "failed_logins_24h": v}
        for i, v in enumerate([0, 1, 2, 3, 10])
    ])
    df = build_model_frame(txns, single_account)
    assert set(df["login_pressure"].astype(str)) <= {"none", "low", "high"}
