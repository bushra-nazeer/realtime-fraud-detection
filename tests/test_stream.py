from fraud.config import load_config
from fraud.features import build_feature_matrix
from fraud.generator import generate_transactions
from fraud.stream.scorer import StreamScorer
from fraud.stream.source import InMemorySource


def test_in_memory_source_is_fifo():
    source = InMemorySource()
    source.produce({"a": 1})
    source.produce({"a": 2})
    assert [m["a"] for m in source.consume()] == [1, 2]


def test_stream_scorer_scores_a_transaction():
    from xgboost import XGBClassifier

    cfg = load_config()
    df = generate_transactions(cfg, n=3000, seed=5)
    X, y = build_feature_matrix(df, cfg.generator.merchant_categories)
    model = XGBClassifier(n_estimators=40, max_depth=4, eval_metric="aucpr", tree_method="hist")
    model.fit(X, y)
    metadata = {
        "feature_names": list(X.columns),
        "merchant_categories": list(cfg.generator.merchant_categories),
        "metrics": {"threshold": 0.5},
    }
    scorer = StreamScorer(cfg, model=model, metadata=metadata)

    result = scorer.score(
        {
            "transaction_id": 1, "customer_id": 7, "amount": 9000.0, "hour": 3,
            "merchant_category": "online", "is_foreign": 1,
            "distance_from_home_km": 800.0, "timestamp": 1.0,
        }
    )
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert {"transaction_id", "fraud_probability", "is_flagged"} <= set(result)
