# Pvlseon

Proof-of-concept dashboard for the take-home brief: upload structured business data, compute KPI trends, visualize them, and generate business-friendly AI commentary.

## Why this stack

This version uses Streamlit, Pandas, Plotly, and optional Claude commentary. Streamlit keeps the demo shippable within a short take-home window while still showing the full data-to-insight workflow the rubric rewards most: ingestion, KPI accuracy, visualization clarity, and narrative interpretation.

## Demo Video

You can watch a quick demo video here to see how the process works, https://www.loom.com/share/dd005ae4a69a447eb25c083c55cea330 

## Tech stack

- Python for the application and KPI logic
- Streamlit for the interactive dashboard UI
- Pandas for CSV loading, schema normalization, and KPI aggregation
- Plotly for interactive trend charts
- Anthropic Claude API for optional AI Insights and AI Summary generation
- python-dotenv for local `.env` key loading
- ReportLab and Kaleido for PDF export with KPI cards, chart images, and AI text
- Pytest for focused KPI logic tests

## Architecture

Pvlseon is split into a small Streamlit UI layer, reusable KPI computation logic, and optional AI commentary generation.

```text
CSV upload or demo data
        |
        v
app.py loads data and collects column mappings
        |
        v
kpi_logic.py detects schema, computes monthly KPIs, and formats dashboard context
        |
        v
app.py renders KPI cards, Plotly trend charts, AI sections, and PDF export
        |
        v
ai_commentary.py calls Claude when a key is available, otherwise returns local fallback commentary
```

The full system diagram is available in [`docs/ARCHITECTURE.mmd`](docs/ARCHITECTURE.mmd), with a rendered image at [`images/dashboard_architecture.png`](images/dashboard_architecture.png).

## Features

- CSV upload with included Superstore-style demo datasets
- Editable column mapping for date, order ID, sales, quantity, profit, and discount columns
- KPI cards for revenue, orders, profit, profit margin, average order value, units sold, and average discount
- Interactive Plotly trend charts
- Optional drill-down by detected dimensions such as region or product line
- AI Insights section with narrative commentary per KPI, plus a separate AI Summary paragraph
- Claude commentary when `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` is configured, with deterministic fallback commentary for no-key demos
- AI generation is button-triggered, so refreshes and filter changes do not repeatedly call Claude
- PDF export with KPI cards, chart images, AI Insights, and AI Summary text

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Claude configuration

The app works without an API key by using local fallback commentary. To call Claude, the person running the demo must provide their own Anthropic API key. For local development, create a `.env` file:

```bash
ANTHROPIC_API_KEY=your_key_here
```

If you already use `CLAUDE_API_KEY` in another local project, this app accepts that name too:

```bash
CLAUDE_API_KEY=your_key_here
```

You can also set it directly before starting Streamlit:

```bash
set ANTHROPIC_API_KEY=your_key_here
```

For Streamlit Cloud, add the same key under app secrets:

```toml
ANTHROPIC_API_KEY = "your_key_here"
```

In the sidebar, leave "Use Claude commentary when API key is configured" enabled. If a key is detected, the commentary section uses Claude. If no key is detected, the app clearly labels the local fallback commentary.

## Commentary prompt logic

Claude receives a compact JSON payload containing the latest month, prior period, and recent KPI history. The system instruction asks it to write for operations leadership, avoid unsupported causes, and return JSON with these keys:

```json
{
  "revenue": "2 to 3 sentence insight",
  "orders": "2 to 3 sentence insight",
  "profit": "2 to 3 sentence insight",
  "profit_margin": "2 to 3 sentence insight",
  "ai_summary": "one paragraph, no more than 8 sentences, with practical next-step insight"
}
```

The app always computes KPI facts locally first. Claude is used for interpretation, not arithmetic, which keeps the demo auditable during the interview.

## Demo data

Original dataset sourced from: `https://www.kaggle.com/datasets/divyjn28/superstore-dataset`, included as `super_store_regular.csv`.

Synthetic versions in the rest of the `data/` folder resemble the same structure for testing across domains.

The demo datasets use a common business schema:

- `Order Date`
- `Order ID`
- `Region`
- `Segment`
- `Category`
- `Sub-Category`
- `Product Name`
- `Sales`
- `Quantity`
- `Discount`
- `Profit`

You can replace it with any public sales, marketing, or operations CSV. The app will try to detect the important columns, then you can correct the mapping in the dashboard before the KPI logic runs.

## Assumptions and limitations

- KPI detection is name-based, but the dashboard includes editable mapping controls for unusual column names.
- Order count is computed from unique `Order ID` when available, otherwise from the mapped quantity/order column.
- Profit margin is computed as `Profit / Sales` when a profit column exists.
- The fallback commentary is deterministic and demo-friendly, but it is labeled as fallback mode. Live Claude output will be richer when an API key is configured.
- Forecasting, email delivery, and saved dashboard configurations are noted as next-step enhancements rather than core scope.

## Submission assets

- Working demo: run with `streamlit run app.py` or deploy to Streamlit Cloud.
- README: this file.
- Demo data: the CSV files in `data/`.
- Architecture diagram: `docs/ARCHITECTURE.mmd` and `images/dashboard_architecture.png`.
- Five-slide outline: `docs/SLIDE_DECK_OUTLINE.md`.
