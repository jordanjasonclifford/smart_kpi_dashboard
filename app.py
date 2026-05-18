from __future__ import annotations

import base64
import hashlib
import re
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from ai_commentary import generate_ai_analysis, has_anthropic_api_key
from kpi_logic import (
    KPI_SPECS,
    Schema,
    build_kpi_context,
    compute_kpis,
    detect_schema,
    format_delta,
    format_metric,
)


APP_TITLE = "Pvlseon"
DATA_DIR = Path("data")
DEFAULT_DEMO_DATASET = "super_store_regular.csv"
LOGO_PATH = Path("pvlseon_kpi_logo_dark_final.svg")


# Configure the Streamlit page before anything gets rendered.
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    # App styling keeps the dark shell and white KPI/insight cards balanced.
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #637083;
            --line: #dfe5ee;
            --panel: #ffffff;
            --accent: #246bfe;
            --good: #0f8a5f;
            --bad: #c43f3f;
            --app-bg: #0e1117;
            --chart-grid: #263040;
            --chart-text: #d7deea;
        }
        .block-container {
            padding-top: 4rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }
        .brand-header {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin: 1.25rem 0 2rem;
        }
        .brand-logo {
            padding-top: 18px;
            padding-left: 2px;
        }
        .brand-logo img {
            display: block;
            width: 640px;
            max-width: 100%;
            height: auto;
        }
        h1 {
            margin-bottom: 0.25rem;
        }
        h2, h3 {
            margin-top: 1.6rem;
        }
        .hero-subtitle {
            color: #aab6c7;
            font-size: 1rem;
            max-width: 760px;
            margin-bottom: 1rem;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 16px 0 22px;
        }
        .status-pill {
            border: 1px solid #2a3445;
            background: #151a24;
            border-radius: 8px;
            padding: 12px 14px;
        }
        .status-label {
            color: #8e9bad;
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .status-value {
            color: #f3f6fb;
            font-size: 1rem;
            font-weight: 750;
            overflow-wrap: anywhere;
        }
        .kpi-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-top: 4px solid var(--accent);
            border-radius: 8px;
            padding: 20px 18px 18px;
            min-height: 148px;
            box-shadow: 0 10px 30px rgba(28, 39, 66, 0.06);
            color: var(--ink);
            margin-bottom: 22px;
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 8px;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 750;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .kpi-delta-badge {
            display: inline-block;
            color: #0f5132;
            background: #d9f7e8;
            border: 1px solid #a9e8c7;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 650;
            padding: 4px 8px;
            line-height: 1.2;
        }
        .kpi-delta-badge.negative {
            color: #842029;
            background: #f8d7da;
            border-color: #f1aeb5;
        }
        .kpi-delta-badge.neutral {
            color: var(--muted);
            background: #eef2f7;
            border-color: var(--line);
        }
        .kpi-delta {
            font-size: 0.92rem;
        }
        .insight-box {
            border-left: 4px solid var(--accent);
            background: #f7f9fd;
            border-radius: 6px;
            padding: 14px 16px;
            margin-bottom: 12px;
            color: var(--ink);
            line-height: 1.45;
        }
        .insight-box strong {
            color: var(--ink);
            display: block;
            margin-bottom: 4px;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stMarkdownContainer"] .kpi-card *,
        div[data-testid="stMarkdownContainer"] .insight-box * {
            color: inherit;
        }
        @media (max-width: 900px) {
            .brand-header {
                align-items: flex-start;
                flex-direction: column;
                gap: 12px;
            }
            .brand-logo img {
                width: min(640px, 92vw);
            }
            .status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv_flexible(source) -> pd.DataFrame:
    # Reads uploaded or local CSVs while handling the encodings used by the demo files.
    # The original Superstore-style data can include Latin-1 characters, so utf-8 alone is not safe.
    last_error: Exception | None = None
    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV with supported encodings: {last_error}")


def list_demo_datasets() -> list[str]:
    # Finds built-in CSV demos and keeps the main Superstore file first.
    # The rest of the files are still available from the sidebar for quick scenario switching.
    if not DATA_DIR.exists():
        return []
    datasets = sorted(path.name for path in DATA_DIR.glob("*.csv") if path.name != "sample_kpi_data.csv")
    if DEFAULT_DEMO_DATASET in datasets:
        datasets.remove(DEFAULT_DEMO_DATASET)
        datasets.insert(0, DEFAULT_DEMO_DATASET)
    return datasets


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file, demo_dataset: str | None) -> pd.DataFrame:
    # Uses an uploaded CSV when present, otherwise falls back to the selected demo dataset.
    if uploaded_file is not None:
        return read_csv_flexible(uploaded_file)
    if demo_dataset:
        return read_csv_flexible(DATA_DIR / demo_dataset)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def cached_compute(raw: pd.DataFrame, group_by: str | None, schema_overrides: dict) -> dict:
    # Caches KPI computation so filter changes do not keep recomputing the same result.
    # The raw dataframe and mapping are part of the cache key, so changed data still recalculates.
    return compute_kpis(raw, group_by=group_by, schema_overrides=schema_overrides)


@st.cache_data(show_spinner=False)
def cached_ai_analysis(context: dict, model: str, enabled: bool, key_available: bool) -> dict[str, str]:
    # Caches Claude or fallback output after the user explicitly generates insights.
    # The UI button controls when this function is called, which avoids surprise API usage.
    return generate_ai_analysis(context=context, model=model, enabled=enabled)


def df_fingerprint(df: pd.DataFrame) -> str:
    # Creates a short stable ID so the app knows when the dataset changed.
    # Only the first 500 rows are used because this is meant as a UI/session signature, not security.
    sample = df.head(500).to_csv(index=False)
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()[:12]


def safe_export_slug(value: str) -> str:
    # Keeps downloaded report names readable while avoiding weird filename characters.
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:60] or "dataset"


def selected_dataset_name(uploaded_file, demo_dataset: str | None) -> str:
    # Uses the real CSV filename in reports so exports can be matched back to the dataset.
    if uploaded_file is not None:
        return uploaded_file.name
    return demo_dataset or "uploaded_dataset.csv"


def analysis_signature(dataset_id: str, schema_overrides: dict, group_by: str | None, model: str) -> str:
    # Tracks the inputs that should invalidate old AI insights.
    # If any of these values change, the old commentary may no longer match the charts.
    payload = {
        "dataset_id": dataset_id,
        "schema_overrides": schema_overrides,
        "group_by": group_by,
        "model": model,
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def render_status_band(items: list[tuple[str, str]]) -> None:
    # Renders compact dataset metadata near the top of the dashboard.
    # This replaces a bulky overview block and makes the loaded data context easy to scan.
    html = "<div class='status-grid'>"
    for label, value in items:
        html += (
            "<div class='status-pill'>"
            f"<div class='status-label'>{label}</div>"
            f"<div class='status-value'>{value}</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_header() -> None:
    # Loads the logo as a data URI so Streamlit renders it as one clean image.
    # Embedding the raw SVG directly caused the wordmark to fight the page styling.
    logo_src = ""
    if LOGO_PATH.exists():
        encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_src = f"data:image/svg+xml;base64,{encoded_logo}"
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-logo">
                <img src="{logo_src}" alt="{APP_TITLE} logo">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def column_options(df: pd.DataFrame, include_none: bool = False) -> list[str]:
    # Builds selectbox options from whatever CSV columns are available.
    options = list(df.columns)
    return ["None"] + options if include_none else options


def option_index(options: list[str], value: str | None) -> int:
    # Picks the detected column in a selectbox when it exists.
    if value and value in options:
        return options.index(value)
    return 0


def infer_initial_schema(raw: pd.DataFrame) -> Schema:
    # Tries normal schema detection first.
    # If that fails, makes a reasonable first guess so the user can fix mapping manually.
    # This keeps the app usable even when someone uploads a CSV with strange column names.
    try:
        return detect_schema(raw)
    except ValueError:
        columns = list(raw.columns)
        numeric_columns = [
            col for col in columns if pd.to_numeric(raw[col], errors="coerce").notna().sum() > 0
        ]
        date_column = columns[0]
        revenue_column = numeric_columns[0] if numeric_columns else columns[min(1, len(columns) - 1)]
        dimension_columns = [
            col
            for col in columns
            if col not in {date_column, revenue_column} and raw[col].nunique() <= 30
        ]
        return Schema(
            date_column=date_column,
            revenue_column=revenue_column,
            profit_column=None,
            order_id_column=None,
            order_column=None,
            discount_column=None,
            sessions_column=None,
            customer_column=None,
            churned_column=None,
            active_column=None,
            dimension_columns=dimension_columns,
        )


def render_column_mapping(raw: pd.DataFrame, schema) -> dict[str, str | None]:
    # Lets the user correct the auto-detected schema without editing code.
    # It is collapsed by default because mapping is setup work, not the main dashboard story.
    required_options = column_options(raw)
    optional_options = column_options(raw, include_none=True)

    with st.expander("Column Mapping", expanded=False):
        st.caption("Auto-detected mappings are editable, so the app can run against different CSV schemas.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            date_column = st.selectbox(
                "Date column",
                required_options,
                index=option_index(required_options, schema.date_column),
            )
            order_id_column = st.selectbox(
                "Order ID column",
                optional_options,
                index=option_index(optional_options, schema.order_id_column),
            )
        with c2:
            revenue_column = st.selectbox(
                "Sales / revenue column",
                required_options,
                index=option_index(required_options, schema.revenue_column),
            )
            profit_column = st.selectbox(
                "Profit column",
                optional_options,
                index=option_index(optional_options, schema.profit_column),
            )
        with c3:
            order_column = st.selectbox(
                "Quantity / units column",
                optional_options,
                index=option_index(optional_options, schema.order_column),
            )
        with c4:
            discount_column = st.selectbox(
                "Discount column",
                optional_options,
                index=option_index(optional_options, schema.discount_column),
            )

    def clean(value: str) -> str | None:
        # Streamlit selectboxes need a string option, but the KPI code wants None.
        return None if value == "None" else value

    return {
        "date_column": date_column,
        "revenue_column": revenue_column,
        "profit_column": clean(profit_column),
        "order_id_column": clean(order_id_column),
        "order_column": clean(order_column),
        "discount_column": clean(discount_column),
        "sessions_column": None,
        "churned_column": None,
        "active_column": None,
    }


def render_kpi_cards(latest: dict, previous: dict | None) -> None:
    # Shows the main KPI facts once at the top of the dashboard.
    # AI sections below interpret these same numbers instead of repeating the cards.
    ordered = [
        "revenue",
        "orders",
        "profit",
        "profit_margin",
        "avg_order_value",
        "units_sold",
        "avg_discount",
    ]
    columns = st.columns(4)

    for index, key in enumerate(ordered):
        spec = KPI_SPECS[key]
        value = latest.get(key)
        prev_value = previous.get(key) if previous else None
        delta = format_delta(value, prev_value, spec["kind"])
        # Color the delta badge based on whether the metric moved up or down.
        delta_class = "neutral"
        if delta.startswith("up"):
            delta_class = "positive"
        elif delta.startswith("down"):
            delta_class = "negative"
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{spec["label"]}</div>
                    <div class="kpi-value">{format_metric(value, spec["kind"])}</div>
                    <div class="kpi-delta-badge {delta_class}">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def style_chart(fig, yaxis_tickprefix: str | None = None, yaxis_tickformat: str | None = None):
    # Applies one Plotly theme so all charts feel like the same dashboard.
    # Bar traces and line traces support different style properties, so trace-specific styling stays light.
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d7deea",
        title_font_size=15,
        title_font_color="#f3f6fb",
        margin=dict(l=10, r=10, t=48, b=20),
        hovermode="x unified",
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#263040", zerolinecolor="#263040", title_font_color="#aab6c7")
    fig.update_yaxes(
        gridcolor="#263040",
        zerolinecolor="#263040",
        title_font_color="#aab6c7",
        tickprefix=yaxis_tickprefix,
        tickformat=yaxis_tickformat,
    )
    fig.update_traces(line_width=2.4, marker_size=6, selector=dict(type="scatter"))
    return fig


def build_chart_figures(monthly: pd.DataFrame, group_by: str | None) -> list:
    # Creates the chart figures once so the page and PDF export use the same visuals.
    # When a drill-down is selected, Plotly uses it as the color dimension for comparison.
    color_arg = group_by if group_by and group_by in monthly.columns else None
    figures = []
    revenue_fig = px.line(
        monthly,
        x="period",
        y="revenue",
        color=color_arg,
        markers=True,
        title="Revenue Trend",
    )
    figures.append(style_chart(revenue_fig, yaxis_tickprefix="$"))

    profit_fig = px.line(
        monthly,
        x="period",
        y="profit",
        color=color_arg,
        markers=True,
        title="Profit Trend",
    )
    figures.append(style_chart(profit_fig, yaxis_tickprefix="$"))

    orders_fig = px.bar(
        monthly,
        x="period",
        y="orders",
        color=color_arg,
        title="Orders by Period",
    )
    orders_fig = style_chart(orders_fig)
    orders_fig.update_traces(marker_color="#7cc7ff")
    figures.append(orders_fig)

    margin_fig = px.line(
        monthly,
        x="period",
        y="profit_margin",
        color=color_arg,
        markers=True,
        title="Profit Margin Trend",
    )
    figures.append(style_chart(margin_fig, yaxis_tickformat=".1%"))
    return figures


def render_charts(figures: list) -> None:
    # Renders the trend explorer from prebuilt figures.
    # The same figure list is later exported into the PDF as chart images.
    for index in range(0, len(figures), 2):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(figures[index], width="stretch")
        if index + 1 < len(figures):
            with right:
                st.plotly_chart(figures[index + 1], width="stretch")


def build_pdf_report(
    latest: dict,
    previous: dict | None,
    figures: list,
    analysis: dict[str, str],
    dataset_id: str,
    dataset_name: str,
) -> bytes:
    # Builds the PDF deliverable shown in the bottom download button.
    # The PDF includes KPI cards, chart images, AI Insights, and the AI Summary.
    import tempfile

    import plotly.io as pio
    from plotly.graph_objects import Figure
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"{APP_TITLE} KPI Report",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#172033"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#172033"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#637083"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardValue",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#172033"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#637083"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="InsightText",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#172033"),
        )
    )

    story = [
        Paragraph(f"{APP_TITLE} KPI Report", styles["ReportTitle"]),
        Paragraph(f"CSV: {escape(dataset_name)}", styles["SmallText"]),
        Paragraph(f"Dataset ID: {dataset_id}", styles["SmallText"]),
        Spacer(1, 0.12 * inch),
        Paragraph("KPI Cards", styles["SectionHeading"]),
    ]

    card_cells = []
    for key in [
        "revenue",
        "orders",
        "profit",
        "profit_margin",
        "avg_order_value",
        "units_sold",
        "avg_discount",
    ]:
        spec = KPI_SPECS[key]
        value = latest.get(key)
        prev_value = previous.get(key) if previous else None
        delta = format_delta(value, prev_value, spec["kind"])
        card_cells.append(
            [
                Paragraph(escape(spec["label"].upper()), styles["CardLabel"]),
                Paragraph(escape(format_metric(value, spec["kind"])), styles["CardValue"]),
                Paragraph(escape(delta), styles["SmallText"]),
            ]
        )

    # ReportLab tables want a rectangular grid, so the final blank cell keeps the card layout even.
    card_cells.append([Paragraph("", styles["SmallText"])])
    card_rows = [
        [card_cells[0], card_cells[1], card_cells[2], card_cells[3]],
        [card_cells[4], card_cells[5], card_cells[6], card_cells[7]],
    ]
    card_table = Table(card_rows, colWidths=[1.82 * inch] * 4, rowHeights=[0.72 * inch] * 2)
    card_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fd")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#dfe5ee")),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#dfe5ee")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(card_table)
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("Trend Images", styles["SectionHeading"]))
    with tempfile.TemporaryDirectory() as temp_dir:
        # Kaleido converts Plotly charts to PNG so the PDF has real chart images.
        # Exporting the charts in one batch is much faster than starting the image engine per chart.
        image_paths = [Path(temp_dir) / f"chart_{index}.png" for index, _ in enumerate(figures)]
        export_figures = []
        for fig in figures:
            # Make the PDF chart titles heavier than the interactive dashboard titles.
            # The dashboard uses dark Plotly styling, but the PDF page is white.
            # Export copies need dark text so titles and axis labels stay readable.
            export_fig = Figure(fig.to_dict())
            title_text = export_fig.layout.title.text or ""
            export_fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#172033", family="Arial"),
                title=dict(
                    text=f"<b>{title_text}</b>",
                    font=dict(size=24, family="Arial Black", color="#172033"),
                ),
            )
            export_fig.update_xaxes(
                title_font=dict(color="#172033", size=15, family="Arial Black"),
                tickfont=dict(color="#172033", size=11),
                gridcolor="#d9e0ea",
                zerolinecolor="#d9e0ea",
                linecolor="#6b7280",
            )
            export_fig.update_yaxes(
                title_font=dict(color="#172033", size=15, family="Arial Black"),
                tickfont=dict(color="#172033", size=11),
                gridcolor="#d9e0ea",
                zerolinecolor="#d9e0ea",
                linecolor="#6b7280",
            )
            export_fig.update_traces(line_color="#111827", marker_color="#111827", selector=dict(type="scatter"))
            export_fig.update_traces(marker_color="#246bfe", selector=dict(type="bar"))
            export_figures.append(export_fig)
        pio.write_images(
            export_figures,
            image_paths,
            format="png",
            width=900,
            height=440,
            scale=2,
        )
        for image_path in image_paths:
            image = Image(str(image_path), width=7.25 * inch, height=3.55 * inch)
            story.extend([image, Spacer(1, 0.12 * inch)])

        story.append(Paragraph("AI Insights", styles["SectionHeading"]))
        for key in ["revenue", "orders", "profit", "profit_margin"]:
            if key not in analysis:
                continue
            insight = KeepTogether(
                [
                    Paragraph(f"<b>{escape(KPI_SPECS[key]['label'])}</b>", styles["InsightText"]),
                    Paragraph(escape(analysis[key]), styles["InsightText"]),
                    Spacer(1, 0.08 * inch),
                ]
            )
            story.append(insight)

        if analysis.get("ai_summary"):
            story.append(Paragraph("AI Summary", styles["SectionHeading"]))
            story.append(Paragraph(escape(analysis["ai_summary"]), styles["InsightText"]))

        doc.build(story)
    return buffer.getvalue()


def render_ai_insights(analysis: dict[str, str]) -> None:
    # Renders the per-KPI AI commentary cards.
    # These are the focused explanations for each metric, not the overall executive summary.
    for key in ["revenue", "orders", "profit", "profit_margin"]:
        if key in analysis:
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>{KPI_SPECS[key]["label"]}</strong>
                    {analysis[key]}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ai_summary(analysis: dict[str, str]) -> None:
    # Renders the single synthesized AI summary paragraph.
    # Claude is asked to keep this to one paragraph so it works as a leadership takeaway.
    ai_summary = analysis.get("ai_summary")
    if not ai_summary:
        return
    st.subheader("AI Summary")
    st.markdown(
        f"""
        <div class="insight-box">
            {ai_summary}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # Main Streamlit flow from data loading to KPIs, charts, and AI commentary.
    # Streamlit reruns this script on interaction, so session state controls generated AI output.
    _inject_styles()

    render_header()

    with st.sidebar:
        # Sidebar contains controls that affect the whole dashboard.
        # The main page stays focused on the analysis output.
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        demo_datasets = list_demo_datasets()
        demo_dataset = None
        if uploaded is None and demo_datasets:
            demo_dataset = st.selectbox("Built-in demo dataset", demo_datasets)
        elif uploaded is not None:
            st.caption("Using uploaded CSV.")

        st.header("Claude commentary")
        key_available = has_anthropic_api_key()
        enable_ai = st.toggle("Use Claude commentary when API key is configured", value=True)
        model = st.text_input(
            "Claude model",
            value="claude-sonnet-4-20250514",
        )
        if key_available:
            st.success("Anthropic API key detected.")
        else:
            st.warning("No Anthropic API key detected. Local fallback commentary will be shown.")
        st.markdown(
            "<span class='small-muted'>Provide ANTHROPIC_API_KEY as an environment variable or Streamlit secret. Without a key, the app uses local demo commentary.</span>",
            unsafe_allow_html=True,
        )

    raw = load_csv(uploaded, demo_dataset)
    if raw.empty:
        st.info("Upload a CSV or choose a built-in demo dataset to begin.")
        return

    dataset_name = selected_dataset_name(uploaded, demo_dataset)

    # Detect columns, then let the user override them in the mapping expander.
    schema = infer_initial_schema(raw)
    schema_overrides = render_column_mapping(raw, schema)
    schema = detect_schema(raw, overrides=schema_overrides)
    group_options = ["None"] + schema.dimension_columns
    dataset_id = df_fingerprint(raw)

    # Show the resolved mapping after overrides so the user knows what the app is using.
    render_status_band(
        [
            ("Rows Loaded", f"{len(raw):,}"),
            ("Date Column", schema.date_column),
            ("Revenue Column", schema.revenue_column),
            ("Dataset ID", dataset_id),
        ]
    )

    group_choice = st.selectbox("Optional drill-down", group_options)
    group_by = None if group_choice == "None" else group_choice

    # Compute all KPI tables from the selected dataset and mapping.
    result = cached_compute(raw, group_by, schema_overrides)
    latest = result["latest"]
    previous = result["previous"]
    monthly = result["monthly"]

    # The dashboard reads top to bottom: facts first, trends second, AI interpretation third.
    st.subheader("Top KPIs")
    render_kpi_cards(latest, previous)

    st.subheader("Trend Explorer")
    figures = build_chart_figures(monthly, group_by)
    render_charts(figures)

    st.subheader("AI Insights")
    context = build_kpi_context(result)

    # Clear old AI output when the inputs behind it are no longer the same.
    current_signature = analysis_signature(dataset_id, schema_overrides, group_by, model)
    if st.session_state.get("analysis_signature") != current_signature:
        st.session_state.pop("ai_analysis", None)
        st.session_state.pop("pdf_report", None)
        st.session_state["analysis_signature"] = current_signature

    st.caption(
        "Generate insights when ready. Results stay on screen until the dataset, column mapping, drill-down, or model changes."
    )
    generate_label = "Generate Claude Insights and Summary" if enable_ai and key_available else "Generate Demo Insights"
    if st.button(generate_label, type="primary"):
        # This is the only place the app calls Claude or creates fallback commentary.
        # Keeping it behind a button makes refreshes safe and keeps API usage intentional.
        with st.spinner("Generating AI insights..."):
            st.session_state.pop("pdf_report", None)
            st.session_state["ai_analysis"] = cached_ai_analysis(
                context,
                model=model,
                enabled=enable_ai,
                key_available=key_available,
            )

    analysis = st.session_state.get("ai_analysis")
    if not analysis:
        # Keep the data preview available even before commentary is generated.
        # This lets someone inspect the CSV before spending an API call.
        st.info("Use Radio Switch Button on sidebar to switch between AI and Demo modes, make sure your API key is configured.")
        st.subheader("Data Preview")
        st.dataframe(raw.head(100), width="stretch")
        return

    if enable_ai and key_available:
        st.caption("Claude-generated commentary based on the KPI context.")
    elif enable_ai:
        st.caption("Local fallback commentary shown because no Anthropic API key is configured.")
    else:
        st.caption("Local fallback commentary shown because Claude commentary is turned off.")
    render_ai_insights(analysis)
    render_ai_summary(analysis)

    st.subheader("Data Preview")
    st.dataframe(raw.head(100), width="stretch")

    if st.button("Prepare KPI PDF"):
        try:
            # This PDF is the take-home friendly artifact from the dashboard.
            # It packages the KPI cards, chart images, and generated commentary into one file.
            with st.spinner("Preparing PDF export..."):
                st.session_state["pdf_report"] = build_pdf_report(
                    latest=latest,
                    previous=previous,
                    figures=figures,
                    analysis=analysis,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                )
        except Exception as exc:
            st.session_state.pop("pdf_report", None)
            st.error(
                "PDF export needs reportlab and kaleido installed. Run pip install -r requirements.txt, then restart Streamlit."
            )
            st.caption(f"Export error: {exc}")

    if st.session_state.get("pdf_report"):
        export_slug = safe_export_slug(Path(dataset_name).stem)
        st.download_button(
            "Download KPI PDF",
            data=st.session_state["pdf_report"],
            file_name=f"pvlseon_{export_slug}_{dataset_id}_kpi_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
