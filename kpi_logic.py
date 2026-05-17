from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


KPI_SPECS = {
    "revenue": {"label": "Revenue", "kind": "currency"},
    "orders": {"label": "Orders", "kind": "integer"},
    "profit": {"label": "Profit", "kind": "currency"},
    "profit_margin": {"label": "Profit Margin", "kind": "percent"},
    "avg_order_value": {"label": "Avg Order Value", "kind": "currency"},
    "units_sold": {"label": "Units Sold", "kind": "integer"},
    "avg_discount": {"label": "Avg Discount", "kind": "percent"},
}


@dataclass(frozen=True)
class Schema:
    date_column: str
    revenue_column: str
    profit_column: str | None
    order_id_column: str | None
    order_column: str | None
    discount_column: str | None
    sessions_column: str | None
    customer_column: str | None
    churned_column: str | None
    active_column: str | None
    dimension_columns: list[str]


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {col.lower().strip().replace(" ", "_"): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for key, original in normalized.items():
        if any(candidate in key for candidate in candidates):
            return original
    return None


def detect_schema(df: pd.DataFrame, overrides: dict[str, str | None] | None = None) -> Schema:
    columns = list(df.columns)
    overrides = overrides or {}

    date_column = overrides.get("date_column") or _find_column(
        columns, ["date", "period", "month", "week", "order_date"]
    )
    if date_column is None:
        raise ValueError("Could not detect a date column. Include a column named date, period, month, or week.")

    revenue_column = overrides.get("revenue_column") or _find_column(
        columns, ["revenue", "sales", "amount", "gross_sales", "net_revenue"]
    )
    if revenue_column is None:
        raise ValueError("Could not detect a revenue column. Include a column named revenue, sales, or amount.")

    profit_column = overrides.get("profit_column") or _find_column(
        columns, ["profit", "gross_profit", "net_profit"]
    )
    order_id_column = overrides.get("order_id_column") or _find_column(
        columns, ["order_id", "transaction_id", "invoice_no", "invoice"]
    )
    order_column = overrides.get("order_column") or _find_column(
        columns, ["orders", "order_count", "transactions", "purchases", "quantity", "units"]
    )
    discount_column = overrides.get("discount_column") or _find_column(
        columns, ["discount", "discount_rate", "promo_discount"]
    )
    sessions_column = overrides.get("sessions_column") or _find_column(
        columns, ["sessions", "visits", "traffic", "leads"]
    )
    customer_column = overrides.get("customer_column") or _find_column(
        columns, ["customers", "new_customers", "customer_count"]
    )
    churned_column = overrides.get("churned_column") or _find_column(
        columns, ["churned_customers", "churned", "cancellations"]
    )
    active_column = overrides.get("active_column") or _find_column(
        columns, ["active_customers", "subscribers", "customer_base"]
    )

    known = {
        date_column,
        revenue_column,
        profit_column,
        order_id_column,
        order_column,
        discount_column,
        sessions_column,
        customer_column,
        churned_column,
        active_column,
    }
    dimension_columns = []
    for col in columns:
        if col in known:
            continue
        is_dimension_type = (
            is_object_dtype(df[col])
            or is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
        if is_dimension_type and 1 < df[col].nunique() <= 30:
            dimension_columns.append(col)

    return Schema(
        date_column=date_column,
        revenue_column=revenue_column,
        profit_column=profit_column,
        order_id_column=order_id_column,
        order_column=order_column,
        discount_column=discount_column,
        sessions_column=sessions_column,
        customer_column=customer_column,
        churned_column=churned_column,
        active_column=active_column,
        dimension_columns=dimension_columns,
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: pd.NA})
    return (numerator / denominator).fillna(0)


def compute_kpis(
    df: pd.DataFrame,
    group_by: str | None = None,
    schema_overrides: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    schema = detect_schema(df, overrides=schema_overrides)
    work = df.copy()
    work[schema.date_column] = pd.to_datetime(work[schema.date_column], errors="coerce")
    work = work.dropna(subset=[schema.date_column])
    work["period"] = work[schema.date_column].dt.to_period("M").dt.to_timestamp()

    metric_columns = [
        schema.revenue_column,
        schema.profit_column,
        schema.order_column,
        schema.discount_column,
        schema.sessions_column,
        schema.customer_column,
        schema.churned_column,
        schema.active_column,
    ]
    for column in [col for col in metric_columns if col]:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)

    group_cols = ["period"]
    if group_by and group_by in work.columns:
        group_cols.append(group_by)

    agg_map = {schema.revenue_column: "sum"}
    if schema.profit_column:
        agg_map[schema.profit_column] = "sum"
    if schema.order_column:
        agg_map[schema.order_column] = "sum"
    if schema.discount_column:
        agg_map[schema.discount_column] = "mean"
    if schema.sessions_column:
        agg_map[schema.sessions_column] = "sum"
    if schema.customer_column:
        agg_map[schema.customer_column] = "sum"
    if schema.churned_column:
        agg_map[schema.churned_column] = "sum"
    if schema.active_column:
        agg_map[schema.active_column] = "sum"

    monthly = work.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    monthly = monthly.rename(
        columns={
            schema.revenue_column: "revenue",
            schema.profit_column or "": "profit",
            schema.order_column or "": "units_sold",
            schema.discount_column or "": "avg_discount",
            schema.sessions_column or "": "sessions",
            schema.customer_column or "": "customers",
            schema.churned_column or "": "churned_customers",
            schema.active_column or "": "active_customers",
        }
    )

    if schema.order_id_column:
        order_counts = (
            work.groupby(group_cols, dropna=False)[schema.order_id_column]
            .nunique()
            .reset_index(name="orders")
        )
        monthly = monthly.drop(columns=["orders"], errors="ignore").merge(
            order_counts, on=group_cols, how="left"
        )
    if "orders" not in monthly:
        monthly["orders"] = monthly["revenue"].gt(0).astype(int)
    if "units_sold" not in monthly:
        monthly["units_sold"] = monthly["orders"]
    if "avg_discount" not in monthly:
        monthly["avg_discount"] = 0
    if "profit" not in monthly:
        monthly["profit"] = 0
    if "sessions" not in monthly:
        monthly["sessions"] = monthly["orders"]
    if "churned_customers" not in monthly:
        monthly["churned_customers"] = 0
    if "active_customers" not in monthly:
        monthly["active_customers"] = monthly.get("customers", monthly["orders"])

    monthly["conversion_rate"] = _safe_divide(monthly["orders"], monthly["sessions"])
    monthly["churn_rate"] = _safe_divide(monthly["churned_customers"], monthly["active_customers"])
    monthly["avg_order_value"] = _safe_divide(monthly["revenue"], monthly["orders"])
    monthly["profit_margin"] = _safe_divide(monthly["profit"], monthly["revenue"])
    monthly = monthly.sort_values(group_cols)

    overall = (
        monthly.groupby("period", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            orders=("orders", "sum"),
            profit=("profit", "sum"),
            units_sold=("units_sold", "sum"),
            avg_discount=("avg_discount", "mean"),
            sessions=("sessions", "sum"),
            churned_customers=("churned_customers", "sum"),
            active_customers=("active_customers", "sum"),
        )
        .sort_values("period")
    )
    overall["conversion_rate"] = _safe_divide(overall["orders"], overall["sessions"])
    overall["churn_rate"] = _safe_divide(overall["churned_customers"], overall["active_customers"])
    overall["avg_order_value"] = _safe_divide(overall["revenue"], overall["orders"])
    overall["profit_margin"] = _safe_divide(overall["profit"], overall["revenue"])

    latest = overall.iloc[-1].to_dict()
    previous = overall.iloc[-2].to_dict() if len(overall) > 1 else None

    return {
        "schema": schema,
        "monthly": monthly,
        "overall": overall,
        "latest": latest,
        "previous": previous,
        "group_by": group_by,
    }


def format_metric(value: Any, kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    if kind == "currency":
        return f"${float(value):,.0f}"
    if kind == "percent":
        return f"{float(value):.1%}"
    if kind == "integer":
        return f"{int(value):,}"
    return str(value)


def format_delta(current: Any, previous: Any, kind: str) -> str:
    if previous is None or pd.isna(previous) or float(previous) == 0:
        return "No prior period comparison"
    change = (float(current) - float(previous)) / abs(float(previous))
    direction = "up" if change >= 0 else "down"
    return f"{direction} {abs(change):.1%} vs prior month"


def build_kpi_context(result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall"].copy()
    context_rows = []
    for _, row in overall.tail(6).iterrows():
        context_rows.append(
            {
                "period": row["period"].strftime("%Y-%m"),
                "revenue": round(float(row["revenue"]), 2),
                "orders": int(row["orders"]),
                "profit": round(float(row["profit"]), 2),
                "profit_margin": round(float(row["profit_margin"]), 4),
                "avg_order_value": round(float(row["avg_order_value"]), 2),
                "units_sold": int(row["units_sold"]),
                "avg_discount": round(float(row["avg_discount"]), 4),
            }
        )
    return {
        "latest": {
            key: (round(float(value), 4) if key != "period" else value.strftime("%Y-%m"))
            for key, value in result["latest"].items()
            if key in {
                "period",
                "revenue",
                "orders",
                "profit",
                "profit_margin",
                "avg_order_value",
                "units_sold",
                "avg_discount",
            }
        },
        "previous": result["previous"],
        "history": context_rows,
        "group_by": result["group_by"],
    }
