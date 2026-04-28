from risk_rules import label_risk, score_transaction


def base_tx(**overrides):
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


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    assert score_transaction(base_tx(amount_usd=1200)) >= 25


def test_high_device_risk_increases_score():
    assert score_transaction(base_tx(device_risk_score=85)) >= 25


def test_international_increases_score():
    assert score_transaction(base_tx(is_international=1)) >= 15


def test_high_velocity_increases_score():
    assert score_transaction(base_tx(velocity_24h=8)) >= 20


def test_prior_chargebacks_increase_score():
    assert score_transaction(base_tx(prior_chargebacks=2)) >= 20
    assert score_transaction(base_tx(prior_chargebacks=1)) >= 5


def test_known_fraud_transaction_scores_high():
    # Mirrors transaction 50011: high device risk, international,
    # large amount, high velocity, many failed logins, prior chargeback.
    tx = base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=1,
    )
    assert label_risk(score_transaction(tx)) == "high"


def test_clean_transaction_scores_low():
    tx = base_tx(amount_usd=45, device_risk_score=8)
    assert label_risk(score_transaction(tx)) == "low"
