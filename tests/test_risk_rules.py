import pytest
from risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_tx(**overrides):
    """All signals at zero — only the overridden field contributes."""
    tx = {
        "device_risk_score": 0,
        "is_international": 0,
        "amount_usd": 0,
        "velocity_24h": 0,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# score_transaction — output range
# ---------------------------------------------------------------------------

def test_score_is_never_negative():
    assert score_transaction(base_tx()) == 0


def test_score_never_exceeds_100():
    # All signals maxed out: 25+15+25+20+20+20 = 125 before clamp.
    tx = base_tx(
        device_risk_score=100,
        is_international=1,
        amount_usd=5000,
        velocity_24h=10,
        failed_logins_24h=10,
        prior_chargebacks=5,
    )
    assert score_transaction(tx) == 100


def test_score_is_integer():
    assert isinstance(score_transaction(base_tx(amount_usd=750)), int)


# ---------------------------------------------------------------------------
# score_transaction — device_risk_score signal (exact values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_contribution", [
    (0,  0),
    (39, 0),
    (40, 10),
    (69, 10),
    (70, 25),
    (100, 25),
])
def test_device_risk_score_contribution(score, expected_contribution):
    assert score_transaction(base_tx(device_risk_score=score)) == expected_contribution


# ---------------------------------------------------------------------------
# score_transaction — is_international signal (exact values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_intl,expected_contribution", [
    (0, 0),
    (1, 15),
])
def test_international_contribution(is_intl, expected_contribution):
    assert score_transaction(base_tx(is_international=is_intl)) == expected_contribution


# ---------------------------------------------------------------------------
# score_transaction — amount_usd signal (exact values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected_contribution", [
    (0,      0),
    (499,    0),
    (500,   10),
    (999,   10),
    (1000,  25),
    (9999,  25),
])
def test_amount_contribution(amount, expected_contribution):
    assert score_transaction(base_tx(amount_usd=amount)) == expected_contribution


# ---------------------------------------------------------------------------
# score_transaction — velocity_24h signal (exact values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("velocity,expected_contribution", [
    (0,  0),
    (2,  0),
    (3,  5),
    (5,  5),
    (6, 20),
    (20, 20),
])
def test_velocity_contribution(velocity, expected_contribution):
    assert score_transaction(base_tx(velocity_24h=velocity)) == expected_contribution


# ---------------------------------------------------------------------------
# score_transaction — failed_logins_24h signal (exact values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("logins,expected_contribution", [
    (0,  0),
    (1,  0),
    (2, 10),
    (4, 10),
    (5, 20),
    (10, 20),
])
def test_failed_logins_contribution(logins, expected_contribution):
    assert score_transaction(base_tx(failed_logins_24h=logins)) == expected_contribution


# ---------------------------------------------------------------------------
# score_transaction — prior_chargebacks signal (exact values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chargebacks,expected_contribution", [
    (0,  0),
    (1,  5),
    (2, 20),
    (5, 20),
])
def test_prior_chargebacks_contribution(chargebacks, expected_contribution):
    assert score_transaction(base_tx(prior_chargebacks=chargebacks)) == expected_contribution


# ---------------------------------------------------------------------------
# score_transaction — additive combination
# ---------------------------------------------------------------------------

def test_signals_are_additive():
    # device(10) + international(15) + amount(10) = 35, all else zero
    tx = base_tx(device_risk_score=50, is_international=1, amount_usd=750)
    assert score_transaction(tx) == 35


def test_all_mid_tier_signals():
    # device(10) + intl(15) + amount(10) + velocity(5) + logins(10) + chargebacks(5) = 55
    tx = base_tx(
        device_risk_score=50,
        is_international=1,
        amount_usd=750,
        velocity_24h=4,
        failed_logins_24h=3,
        prior_chargebacks=1,
    )
    assert score_transaction(tx) == 55


def test_all_top_tier_signals_capped_at_100():
    # device(25) + intl(15) + amount(25) + velocity(20) + logins(20) + chargebacks(20) = 125 → 100
    tx = base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=2000,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100


# ---------------------------------------------------------------------------
# label_risk — exact boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_label", [
    (0,   "low"),
    (29,  "low"),
    (30,  "medium"),
    (59,  "medium"),
    (60,  "high"),
    (100, "high"),
])
def test_label_risk_boundaries(score, expected_label):
    assert label_risk(score) == expected_label


# ---------------------------------------------------------------------------
# End-to-end: real-world transaction profiles
# ---------------------------------------------------------------------------

def test_clean_low_risk_transaction_scores_low():
    # Low amount, domestic, no red flags.
    tx = base_tx(amount_usd=45, device_risk_score=8)
    assert label_risk(score_transaction(tx)) == "low"


def test_high_amount_only_scores_low():
    # Large amount adds 25 points, which falls below the medium threshold (30).
    # A big purchase with no other fraud signals is low risk.
    tx = base_tx(amount_usd=1500)
    assert score_transaction(tx) == 25
    assert label_risk(score_transaction(tx)) == "low"


def test_account_takeover_profile_scores_medium():
    # international(15) + velocity(20) + failed_logins(20) = 55 → medium.
    # Without a large amount or device/chargeback signal it falls just short of high (60).
    tx = base_tx(
        is_international=1,
        velocity_24h=8,
        failed_logins_24h=6,
    )
    assert score_transaction(tx) == 55
    assert label_risk(score_transaction(tx)) == "medium"


def test_repeat_fraudster_profile_scores_high():
    # Mirrors transaction 50011 from the dataset (confirmed chargeback, $1,400 loss).
    tx = base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=1,
    )
    assert label_risk(score_transaction(tx)) == "high"
    assert score_transaction(tx) == 100
