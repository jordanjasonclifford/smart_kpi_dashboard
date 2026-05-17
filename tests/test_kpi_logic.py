import pandas as pd

from kpi_logic import build_kpi_context, compute_kpis, detect_schema


def test_detect_schema_on_sample_shape():
    df = pd.read_csv("data/super_store_regular.csv", encoding="latin1")
    schema = detect_schema(df)

    assert schema.date_column == "Order Date"
    assert schema.revenue_column == "Sales"
    assert schema.profit_column == "Profit"
    assert schema.order_id_column == "Order ID"
    assert schema.order_column == "Quantity"
    assert schema.discount_column == "Discount"
    assert "Region" in schema.dimension_columns


def test_compute_kpis_returns_latest_and_history():
    df = pd.read_csv("data/super_store_regular.csv", encoding="latin1")
    result = compute_kpis(df)

    assert result["latest"]["revenue"] != 0
    assert result["latest"]["orders"] > 0
    assert "profit" in result["latest"]
    assert "profit_margin" in result["latest"]
    assert "avg_order_value" in result["latest"]
    assert "units_sold" in result["latest"]
    assert "avg_discount" in result["latest"]

    context = build_kpi_context(result)
    assert len(context["history"]) <= 6
    assert {
        "revenue",
        "orders",
        "profit",
        "profit_margin",
        "avg_order_value",
        "units_sold",
        "avg_discount",
    }.issubset(context["history"][-1])
