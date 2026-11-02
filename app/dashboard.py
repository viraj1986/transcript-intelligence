"""
dashboard.py — Streamlit view of Transcript Intelligence results.

Reads data/results.csv from src/pipeline.py and covers the three assignment asks:
1. Categories, with one example meeting from each
2. Sentiment across call types, and what the trend means
3. Additional signals: outage blast radius, product demand, escalation risk

Usage:
    streamlit run app/dashboard.py

Call type on the charts is reviewed before display. The model sometimes labels
a customer meeting as internal, or a renewal as support. The review uses
attendee email domains, the title, and the category so the charts follow the
assignment's definitions of support, external, and internal. The model's
original label is kept as call_type_model.
"""

import ast
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_CSV = BASE_DIR / "data" / "results.csv"
FAILED_CSV = BASE_DIR / "data" / "failed.csv"

CALL_TYPES = ["support", "external", "internal"]
SENTIMENTS = ["negative", "neutral", "positive"]
CATEGORIES = [
    "Billing",
    "Technical Support",
    "Account Management",
    "Incident Response",
    "Product Feedback",
    "Internal Planning",
    "Onboarding",
    "Other",
]
EXTERNAL_CATEGORIES = {"Account Management", "Onboarding", "Product Feedback"}
SUPPORT_CATEGORIES = {"Technical Support", "Billing", "Incident Response"}

PRODUCT_ALIASES = {
    "aegis detect": "Aegis Detect",
    "detect": "Aegis Detect",
    "aegis protect": "Aegis Protect",
    "protect": "Aegis Protect",
    "aegis identity": "Aegis Identity",
    "identity": "Aegis Identity",
    "aegis sso": "Aegis Identity",
    "comply v2": "Comply v2",
    "aegis comply v2": "Comply v2",
    "aegis comply": "Aegis Comply",
    "comply": "Aegis Comply",
}
PRODUCT_ORDER = [
    "Aegis Detect",
    "Comply v2",
    "Aegis Protect",
    "Aegis Identity",
    "Aegis Comply",
]
TOPIC_ALIASES = {
    "detect outage": "Detect outage",
    "aegis detect outage": "Detect outage",
    "comply v2": "Comply v2",
    "aegis comply v2": "Comply v2",
    "comply v2 launch": "Comply v2",
    "comply v2 ga": "Comply v2",
    "comply v2 on-demand reporting": "Comply v2",
    "hipaa compliance": "HIPAA compliance",
    "hipaa audit": "HIPAA compliance",
    "pci dss": "PCI DSS",
    "pci dss compliance": "PCI DSS",
    "pci dss reporting": "PCI DSS",
    "soc 2": "SOC 2",
    "soc 2 reporting": "SOC 2",
    "iso 27001": "ISO 27001",
    "iso 27001 compliance": "ISO 27001",
    "sentinelshield": "SentinelShield",
    "sentinelshield evaluation": "SentinelShield",
    "aegis protect": "Aegis Protect",
    "incident response": "Incident response",
}
COMPETITORS = ("sentinelshield", "vaultedge", "cybernova", "fortiguard")

SENTIMENT_SCALE = alt.Scale(
    domain=SENTIMENTS,
    range=["#b4534a", "#c4b48a", "#2f6f4e"],
)
RISK_SCALE = alt.Scale(
    domain=["No escalation", "Escalation risk"],
    range=["#d9d3c5", "#b4534a"],
)


def parse_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def canon_category(value):
    key = str(value).strip().lower()
    for name in CATEGORIES:
        if key == name.lower():
            return name
    return str(value).strip()


def canon_sentiment(value):
    key = str(value).strip().lower()
    return key if key in SENTIMENTS else key


def email_domains(attendees):
    domains = set()
    for item in parse_list(attendees):
        if "@" in item:
            domains.add(item.split("@", 1)[1].lower())
    return domains


def review_call_type(attendees, title, category, model_label):
    """Align call type with the assignment definitions.

    Internal means every attendee shares one email domain.
    A support-case title is support even if the model said otherwise.
    Remaining customer meetings follow the category: account, onboarding,
    and feedback calls are external; technical, billing, and incident
    calls are support.
    """
    domains = email_domains(attendees)
    title_l = str(title or "").lower()
    model = str(model_label or "").strip().lower()
    if len(domains) <= 1:
        return "internal"
    if "support case" in title_l or "case #" in title_l:
        return "support"
    if category in EXTERNAL_CATEGORIES:
        return "external"
    if category in SUPPORT_CATEGORIES:
        return "support"
    if model in {"support", "external"}:
        return model
    return "external"


def canonical_products(items):
    found = []
    seen = set()
    for raw in items:
        name = PRODUCT_ALIASES.get(str(raw).strip().lower())
        if name and name not in seen:
            seen.add(name)
            found.append(name)
    return found


def canonical_topics(items):
    found = []
    seen = set()
    for raw in items:
        key = " ".join(str(raw).strip().lower().split())
        name = TOPIC_ALIASES.get(key, str(raw).strip())
        if name and name not in seen:
            seen.add(name)
            found.append(name)
    return found


def mentions_outage(title, primary_topic, topics):
    blob = " ".join([str(title or ""), str(primary_topic or ""), *topics]).lower()
    return "outage" in blob


def mentions_competitor(primary_topic, topics, products):
    blob = " ".join([str(primary_topic or ""), *topics, *products]).lower()
    return any(name in blob for name in COMPETITORS)


def _pct(part, whole):
    if not whole:
        return 0.0
    return 100.0 * part / whole


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    df["secondary_topics"] = df["secondary_topics"].apply(parse_list)
    df["products_mentioned"] = df["products_mentioned"].apply(parse_list)
    df["category"] = df["category"].apply(canon_category)
    df["sentiment_label"] = df["sentiment_label"].apply(canon_sentiment)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["escalation_risk"] = df["escalation_risk"].apply(parse_bool)
    if "is_business_call" in df.columns:
        df["is_business_call"] = df["is_business_call"].apply(parse_bool)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str).where(df["date"].notna(), None)

    df["call_type_model"] = df["call_type"].astype(str).str.strip().str.lower()
    df["call_type"] = [
        review_call_type(attendees, title, category, model)
        for attendees, title, category, model in zip(
            df["attendees"], df["title"], df["category"], df["call_type_model"]
        )
    ]
    df["canonical_products"] = df["products_mentioned"].apply(canonical_products)
    df["canonical_topics"] = df["secondary_topics"].apply(canonical_topics)
    df["mentions_outage"] = [
        mentions_outage(title, topic, topics)
        for title, topic, topics in zip(df["title"], df["primary_topic"], df["secondary_topics"])
    ]
    df["mentions_competitor"] = [
        mentions_competitor(topic, topics, products)
        for topic, topics, products in zip(
            df["primary_topic"], df["secondary_topics"], df["products_mentioned"]
        )
    ]
    return df


@st.cache_data
def load_failed(path: str) -> pd.DataFrame:
    failed_path = Path(path)
    if not failed_path.exists() or failed_path.stat().st_size == 0:
        return pd.DataFrame(columns=["meeting_id", "title"])
    try:
        return pd.read_csv(failed_path, usecols=["meeting_id", "title"])
    except (ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["meeting_id", "title"])


def filter_meetings(df, call_types, categories, sentiments, escalation, months):
    out = df[df["call_type"].isin(call_types)]
    out = out[out["category"].isin(categories)]
    out = out[out["sentiment_label"].isin(sentiments)]
    if escalation == "High risk only":
        out = out[out["escalation_risk"]]
    elif escalation == "Low risk only":
        out = out[~out["escalation_risk"]]
    out = out[out["month"].isin(months)]
    return out.copy()


def transcript_excerpt(meeting_id, max_turns=3, max_chars=220):
    """A few spoken lines from the source transcript, not the model summary."""
    path = BASE_DIR / "data" / "raw" / str(meeting_id) / "transcript.json"
    if not path.exists():
        return ""
    turns = json.loads(path.read_text())["data"]
    start = 0
    for i, turn in enumerate(turns):
        if len(turn["sentence"].split()) >= 12:
            start = i
            break
    lines = []
    for turn in turns[start:start + max_turns]:
        sentence = " ".join(turn["sentence"].split())
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars].rsplit(" ", 1)[0] + "…"
        lines.append(f"{turn['speaker_name']}: {sentence}")
    return "\n".join(lines)


# Title hints keep the sample meeting on-theme. Highest confidence alone
# surfaces long meetings the model labeled confidently but off-theme,
# such as a launch review filed under Technical Support.
CATEGORY_TITLE_HINTS = {
    "Billing": ("billing", "invoice", "overage"),
    "Technical Support": ("support case",),
    "Account Management": ("renewal", "account", "qbr", "business review"),
    "Incident Response": ("outage", "incident", "escalation", "war room"),
    "Product Feedback": ("feedback",),
    "Internal Planning": ("planning", "sprint", "all hands", "standup", "roadmap"),
    "Onboarding": ("onboarding", "kickoff", "deployment"),
}


def category_examples(df):
    rows = []
    for category, sub in df.groupby("category", sort=False):
        hints = CATEGORY_TITLE_HINTS.get(category, ())
        titles = sub["title"].fillna("").str.lower()
        matched = sub[titles.apply(lambda title: any(hint in title for hint in hints))] if hints else sub.iloc[0:0]
        pool = matched if len(matched) else sub
        ranked = pool.sort_values(
            ["confidence_score", "sentence_count"],
            ascending=False,
        )
        rows.append(ranked.iloc[0])
    examples = pd.DataFrame(rows)
    order = df["category"].value_counts().index
    examples = examples.set_index("category").loc[list(order)].reset_index()
    examples["transcript_excerpt"] = examples["meeting_id"].map(transcript_excerpt)
    return examples[
        ["category", "title", "call_type", "sentiment_label", "confidence_score", "transcript_excerpt"]
    ]


def product_monthly(df):
    rows = []
    for _, meeting in df.iterrows():
        for product in meeting["canonical_products"]:
            rows.append({"month": meeting["month"], "product": product, "meeting_id": meeting["meeting_id"]})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    monthly = (
        frame.groupby(["month", "product"])["meeting_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    columns = [product for product in PRODUCT_ORDER if product in monthly.columns]
    return monthly[columns]


def call_mix_note(df):
    counts = df["call_type"].value_counts()
    if counts.empty:
        return "No meetings in this filter."
    parts = [f"{name} {int(counts[name])}" for name in CALL_TYPES if name in counts.index]
    largest = counts.idxmax()
    return (
        f"Call mix in this filter: {', '.join(parts)}. "
        f"**{str(largest).title()}** is the largest group."
    )


def category_note(df):
    counts = df["category"].value_counts()
    if counts.empty:
        return ""
    if len(counts) == 1:
        lead = f"The only theme in this filter is **{counts.index[0]}** ({int(counts.iloc[0])} meetings)."
    else:
        lead = (
            f"The two largest themes are **{counts.index[0]}** ({int(counts.iloc[0])}) "
            f"and **{counts.index[1]}** ({int(counts.iloc[1])})."
        )
    rates = df.groupby("category").agg(
        n=("meeting_id", "size"),
        esc=("escalation_risk", "mean"),
        score=("sentiment_score", "mean"),
    )
    rates = rates[rates["n"] >= 3]
    if rates.empty:
        return lead
    highest = rates["esc"].idxmax()
    warmest = rates["score"].idxmax()
    return (
        f"{lead} **{highest}** has the highest escalation rate among themes with at least "
        f"3 meetings ({rates.loc[highest, 'esc']:.0%}, {int(rates.loc[highest, 'n'])} meetings). "
        f"**{warmest}** has the warmest average sentiment score "
        f"({rates.loc[warmest, 'score']:+.2f})."
    )


def sentiment_note(df):
    lines = []
    for call_type in CALL_TYPES:
        sub = df[df["call_type"] == call_type]
        if sub.empty:
            continue
        negative = int((sub["sentiment_label"] == "negative").sum())
        positive = int((sub["sentiment_label"] == "positive").sum())
        escalated = int(sub["escalation_risk"].sum())
        lines.append(
            f"**{call_type.title()}** — {len(sub)} meetings, {negative} negative, "
            f"{positive} positive, {escalated} escalation-risk "
            f"({_pct(escalated, len(sub)):.0f}%)."
        )
    rates = df.groupby("call_type")["escalation_risk"].mean()
    if not rates.empty:
        top = rates.idxmax()
        lines.append(
            f"Escalation risk is highest on **{top}** calls ({rates[top]:.0%} of those meetings)."
        )
        if top == "support":
            lines.append(
                "That is the staffing signal for support leadership: the risk sits in the customer problem queue."
            )
        elif top == "external":
            lines.append(
                "Account managers are carrying the escalation risk on relationship calls, ahead of the support queue."
            )
        elif top == "internal":
            lines.append(
                "The escalation flags are on internal calls, which here are incident bridges and planning reviews."
            )
    return "\n\n".join(lines)


def score_conflict_note(df):
    negative_high = df[(df["sentiment_label"] == "negative") & (df["sentiment_score"] > 0)]
    positive_low = df[(df["sentiment_label"] == "positive") & (df["sentiment_score"] < 0)]
    conflicts = len(negative_high) + len(positive_low)
    neutral_median = df.loc[df["sentiment_label"] == "neutral", "sentiment_score"].median()
    if conflicts == 0:
        return "In this filter, every negative label has a score at or below zero and every positive label has a score at or above zero."
    neutral_bit = ""
    if pd.notna(neutral_median):
        neutral_bit = f" Neutral labels have a median score of {neutral_median:+.2f}."
    return (
        f"**{conflicts}** meetings have a label and a score on opposite sides of zero "
        f"({len(negative_high)} labeled negative with a score above 0, "
        f"{len(positive_low)} labeled positive with a score below 0). "
        f"The average score therefore reads warmer than the label mix.{neutral_bit} "
        f"Use the label chart for the mix of calls, and the weekly score chart for the shape of the period."
    )


def trend_note(df):
    dated = df.dropna(subset=["date"])
    if dated["month"].nunique() < 2:
        return "This filter covers a single month, so there is no month-to-month trend to read."
    monthly = dated.groupby("month").agg(
        n=("meeting_id", "size"),
        score=("sentiment_score", "mean"),
        neg=("sentiment_label", lambda labels: (labels == "negative").mean()),
        outages=("mentions_outage", "sum"),
    )
    worst = monthly["score"].idxmin()
    row = monthly.loc[worst]
    text = (
        f"The weakest month in this filter is **{worst}**: average sentiment score "
        f"{row['score']:+.2f}, {row['neg']:.0%} of meetings labeled negative, "
        f"and {int(row['outages'])} of {int(row['n'])} meetings mention an outage."
    )
    later = monthly.loc[monthly.index > worst]
    if len(later):
        nxt = later.iloc[0]
        text += (
            f" **{later.index[0]}** averages {nxt['score']:+.2f} "
            f"with {nxt['neg']:.0%} negative."
        )
    if row["outages"] > 0 and row["outages"] == monthly["outages"].max():
        text += (
            " That month also contains the most outage mentions in this filter, "
            "so the drop lines up with the incident window."
        )
    return text


def outage_note(df):
    flagged = df[df["mentions_outage"]]
    if flagged.empty:
        return (
            "No meeting in this filter mentions an outage in the title, primary topic, "
            "or secondary topics."
        )
    counts = flagged["call_type"].value_counts()
    parts = ", ".join(
        f"{int(counts[name])} {name}" for name in CALL_TYPES if name in counts.index
    )
    return (
        f"**{len(flagged)}** meetings mention an outage ({parts}). "
        f"The same incident shows up on support calls, internal bridges, and customer reviews, "
        f"so the blast radius is visible only when those call types are read together. "
        f"Useful for engineering leads, support leaders, and customer success."
    )


def product_note(df, monthly):
    if monthly.empty:
        return "No recognized Aegis product mentions in this filter."
    totals = {}
    for products in df["canonical_products"]:
        for product in products:
            totals[product] = totals.get(product, 0) + 1
    top = max(totals, key=totals.get)
    text = f"**{top}** is mentioned in the most meetings ({totals[top]})."
    if len(monthly) >= 2:
        delta = monthly.iloc[-1] - monthly.iloc[0]
        mover = delta.abs().idxmax()
        start = int(monthly.iloc[0][mover])
        end = int(monthly.iloc[-1][mover])
        if end > start:
            text += (
                f" **{mover}** grows the most from the first month in this filter to the last, "
                f"{start} meetings to {end}."
            )
        elif end < start:
            text += (
                f" **{mover}** falls the most from the first month in this filter to the last, "
                f"{start} meetings to {end}."
            )
        else:
            text += (
                f" Product counts are unchanged from {monthly.index[0]} to {monthly.index[-1]}."
            )
    competitors = int(df["mentions_competitor"].sum())
    if competitors:
        text += (
            f" **{competitors}** meetings name a competitor "
            f"(SentinelShield, VaultEdge, CyberNova, or Fortiguard), "
            f"which is a renewal and displacement signal."
        )
    text += (
        " Each meeting counts once per product. "
        "Detect and Aegis Detect are combined, as are Comply v2 and Aegis Comply v2. "
        "Useful for product managers and customer success."
    )
    return text


def risk_note(df):
    escalated = df[df["escalation_risk"]]
    if escalated.empty:
        return "No escalation-risk meetings in this filter."
    shares = escalated["category"].value_counts()
    top = shares.head(2)
    parts = [f"**{cat}** ({int(n)})" for cat, n in top.items()]
    if len(parts) == 1:
        top_bit = parts[0]
    elif len(parts) == 2:
        top_bit = f"{parts[0]} and {parts[1]}"
    else:
        top_bit = ""
    neutral = int(((df["sentiment_label"] == "neutral") & df["escalation_risk"]).sum())
    verb = "accounts" if len(parts) == 1 else "account"
    text = (
        f"**{len(escalated)}** meetings are escalation-risk. {top_bit} "
        f"{verb} for {int(top.sum())} of them. "
        f"**{neutral}** escalation-risk meetings are labeled neutral, "
        f"so a review that only pulls negative calls misses them."
    )
    calm = df[~df["escalation_risk"]]
    if len(calm) and escalated["duration_min"].notna().any() and calm["duration_min"].notna().any():
        text += (
            f" Escalation-risk meetings average {escalated['duration_min'].mean():.0f} minutes; "
            f"the other meetings average {calm['duration_min'].mean():.0f} minutes."
        )
    long_support = df[(df["call_type"] == "support") & (df["duration_min"] >= 30)]
    support_n = int((df["call_type"] == "support").sum())
    if support_n:
        text += (
            f" {len(long_support)} of {support_n} support meetings run 30 minutes or longer, "
            f"which is the tail a support lead would inspect for stuck issues."
        )
    text += " Useful for support leaders and account managers."
    return text


def hbar(series, title, x_title="Meetings", sort="-x"):
    if series is None or len(series) == 0:
        return None
    frame = series.rename("count").reset_index()
    frame.columns = ["label", "count"]
    return (
        alt.Chart(frame)
        .mark_bar(color="#3d5a80")
        .encode(
            x=alt.X("count:Q", title=x_title),
            y=alt.Y("label:N", sort=sort, title=None),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("count:Q", title=x_title)],
        )
        .properties(title=title, height=max(28 * len(frame) + 30, 140))
    )


def stacked_sentiment(df):
    counts = (
        df.groupby(["call_type", "sentiment_label"])
        .size()
        .reset_index(name="meetings")
    )
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("call_type:N", sort=CALL_TYPES, title=None),
            y=alt.Y("meetings:Q", stack="zero", title="Meetings"),
            color=alt.Color(
                "sentiment_label:N",
                sort=SENTIMENTS,
                scale=SENTIMENT_SCALE,
                title="Sentiment",
            ),
            tooltip=["call_type", "sentiment_label", "meetings"],
        )
        .properties(title="Sentiment by call type", height=280)
    )


def stacked_risk(df):
    plot_df = df.assign(
        risk=df["escalation_risk"].map({True: "Escalation risk", False: "No escalation"})
    )
    counts = plot_df.groupby(["call_type", "risk"]).size().reset_index(name="meetings")
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("call_type:N", sort=CALL_TYPES, title=None),
            y=alt.Y("meetings:Q", stack="zero", title="Meetings"),
            color=alt.Color("risk:N", sort=["No escalation", "Escalation risk"], scale=RISK_SCALE, title=None),
            tooltip=["call_type", "risk", "meetings"],
        )
        .properties(title="Escalation risk by call type", height=280)
    )


def escalation_rate_by_sentiment(df):
    frame = (
        df.groupby("sentiment_label")
        .agg(meetings=("meeting_id", "size"), escalation_risk=("escalation_risk", "sum"))
        .reindex(SENTIMENTS)
        .dropna(subset=["meetings"])
        .reset_index()
    )
    frame["rate"] = frame["escalation_risk"] / frame["meetings"]
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("sentiment_label:N", sort=SENTIMENTS, title=None),
            y=alt.Y("rate:Q", title="Share escalation-risk", axis=alt.Axis(format="%")),
            color=alt.Color("sentiment_label:N", sort=SENTIMENTS, scale=SENTIMENT_SCALE, legend=None),
            tooltip=[
                "sentiment_label",
                "meetings",
                alt.Tooltip("escalation_risk:Q", title="Escalation-risk meetings"),
                alt.Tooltip("rate:Q", title="Share", format=".0%"),
            ],
        )
        .properties(title="Escalation rate by sentiment label", height=280)
    )


def sentiment_trend(df):
    weekly = (
        df.dropna(subset=["date"])
        .groupby(pd.Grouper(key="date", freq="W-SUN"))
        .agg(score=("sentiment_score", "mean"), meetings=("meeting_id", "size"))
        .reset_index()
    )
    weekly = weekly[weekly["meetings"] > 0]
    if weekly.empty:
        return None
    line = (
        alt.Chart(weekly)
        .mark_line(point=True, color="#3d5a80")
        .encode(
            x=alt.X("date:T", title="Week ending"),
            y=alt.Y("score:Q", title="Average sentiment score", scale=alt.Scale(domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("date:T", title="Week ending"),
                alt.Tooltip("score:Q", title="Average score", format="+.2f"),
                "meetings",
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="y:Q")
    return (line + zero).properties(title="Weekly average sentiment score", height=280)


def monthly_summary(df):
    summary = (
        df.dropna(subset=["month"])
        .groupby("month")
        .agg(
            meetings=("meeting_id", "size"),
            avg_sentiment_score=("sentiment_score", "mean"),
            pct_negative=("sentiment_label", lambda labels: 100 * (labels == "negative").mean()),
            escalation_risk=("escalation_risk", "sum"),
            outage_meetings=("mentions_outage", "sum"),
        )
        .reset_index()
    )
    summary["avg_sentiment_score"] = summary["avg_sentiment_score"].round(2)
    summary["pct_negative"] = summary["pct_negative"].round(0)
    return summary


def product_chart(monthly):
    if monthly.empty:
        return None
    frame = monthly.reset_index().melt("month", var_name="product", value_name="meetings")
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("month:N", title=None, sort=None),
            y=alt.Y("meetings:Q", title="Meetings"),
            color=alt.Color("product:N", sort=PRODUCT_ORDER, title=None),
            xOffset=alt.XOffset("product:N", sort=PRODUCT_ORDER),
            tooltip=["month", "product", "meetings"],
        )
        .properties(title="Aegis products mentioned, by month", height=300)
    )


def duration_chart(df):
    rows = []
    for call_type in CALL_TYPES:
        sub = df.loc[df["call_type"] == call_type, "duration_min"].dropna()
        if sub.empty:
            continue
        rows.append({"call_type": call_type, "stat": "Average", "minutes": float(sub.mean())})
        rows.append(
            {"call_type": call_type, "stat": "75th percentile", "minutes": float(sub.quantile(0.75))}
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("call_type:N", sort=CALL_TYPES, title=None),
            y=alt.Y("minutes:Q", title="Minutes"),
            color=alt.Color("stat:N", title=None),
            xOffset="stat:N",
            tooltip=["call_type", "stat", alt.Tooltip("minutes:Q", format=".1f")],
        )
        .properties(title="Meeting length by call type", height=280)
    )


def show_chart(chart):
    if chart is not None:
        st.altair_chart(chart, width="stretch")


def coverage_caption(df_all, failed):
    dated = df_all["date"].dropna()
    if dated.empty:
        span = "undated meetings"
    else:
        span = f"{dated.min():%b %d, %Y} – {dated.max():%b %d, %Y}"
    text = f"{len(df_all)} classified meetings, {span}."
    if len(failed):
        titles = "; ".join(failed["title"].dropna().astype(str).head(3))
        extra = "" if len(failed) <= 3 else f" and {len(failed) - 3} more"
        text += f" {len(failed)} meeting(s) failed validation and are excluded: {titles}{extra}."
    low_conf = int((df_all["confidence_score"] < 0.6).sum())
    conf_min = df_all["confidence_score"].min()
    conf_max = df_all["confidence_score"].max()
    text += (
        f" Model confidence runs from {conf_min:.2f} to {conf_max:.2f}. "
        f"Validation accepts a score from 0 to 1. "
        f"Scores under 0.6 are a review check on this page ({low_conf} meetings). "
        f"High confidence means the model committed, which is separate from the label being right."
    )
    return text


def main():
    st.set_page_config(page_title="Transcript Intelligence", page_icon="📞", layout="wide")

    try:
        df_all = load_results(str(RESULTS_CSV))
    except FileNotFoundError:
        st.error(f"results.csv not found at `{RESULTS_CSV}`. Run `python src/pipeline.py` first.")
        st.stop()

    if df_all.empty:
        st.error("results.csv has no meetings.")
        st.stop()

    failed = load_failed(str(FAILED_CSV))

    st.sidebar.title("Filters")
    call_types = [name for name in CALL_TYPES if name in set(df_all["call_type"])]
    call_types += sorted(set(df_all["call_type"]) - set(CALL_TYPES))
    categories = [name for name in CATEGORIES if name in set(df_all["category"])]
    categories += sorted(set(df_all["category"]) - set(CATEGORIES))
    sentiments = [name for name in SENTIMENTS if name in set(df_all["sentiment_label"])]
    months = [month for month in df_all["month"].dropna().unique()]

    selected_types = st.sidebar.multiselect("Call type", call_types, default=call_types)
    selected_cats = st.sidebar.multiselect("Category", categories, default=categories)
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)
    selected_months = st.sidebar.multiselect("Month", months, default=list(months))
    escalation = st.sidebar.radio("Escalation risk", ["All", "High risk only", "Low risk only"])
    st.sidebar.caption(
        "Call type is the reviewed label. A single email domain is internal. "
        "A support-case title is support. Other customer meetings follow the category."
    )
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

    df = filter_meetings(
        df_all, selected_types, selected_cats, selected_sentiments, escalation, selected_months
    )

    st.title("Transcript Intelligence")
    st.caption(coverage_caption(df_all, failed))
    st.caption(f"Showing {len(df)} of {len(df_all)} meetings after filters.")

    if df.empty:
        st.info("No meetings match the current filters. Select at least one value in each filter.")
        st.stop()

    positive_pct = _pct((df["sentiment_label"] == "positive").sum(), len(df))
    negative_pct = _pct((df["sentiment_label"] == "negative").sum(), len(df))
    escalated_n = int(df["escalation_risk"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Meetings", len(df))
    m2.metric("Positive", f"{positive_pct:.0f}%")
    m3.metric("Negative", f"{negative_pct:.0f}%")
    m4.metric("Escalation risk", f"{escalated_n} ({_pct(escalated_n, len(df)):.0f}%)")

    # ── 1. Categories ────────────────────────────────────────────────────────
    st.header("1. Categories")
    st.markdown(
        "Each meeting gets one theme from the local Llama pipeline: Billing, Technical Support, "
        "Account Management, Incident Response, Product Feedback, Internal Planning, Onboarding, or Other. "
        "Call type says who was on the line. Category says what the call was about."
    )

    left, right = st.columns(2)
    with left:
        show_chart(
            hbar(
                df["call_type"].value_counts().reindex(CALL_TYPES).dropna(),
                "Meetings by call type",
                sort=CALL_TYPES,
            )
        )
        st.caption(
            "Support: a customer is on a product, billing, or incident call. "
            "External: a customer is on a renewal, onboarding, account review, or feedback call. "
            "Internal: every attendee is on one email domain."
        )
    with right:
        show_chart(hbar(df["category"].value_counts(), "Meetings by category"))

    st.markdown(call_mix_note(df))
    st.markdown(category_note(df))

    relabeled = df_all[df_all["call_type"] != df_all["call_type_model"]]
    with st.expander(f"How call type was reviewed ({len(relabeled)} model labels updated)"):
        st.markdown(
            "The model label is kept on every row as `call_type_model`. Charts use a reviewed label. "
            "A single email domain is **internal**. A title with “Support Case” or “Case #” is **support**. "
            "Any other meeting with a customer on the line follows its category: "
            "Account Management, Onboarding, and Product Feedback are **external**; "
            "Technical Support, Billing, and Incident Response are **support**."
        )
        model_counts = df_all["call_type_model"].value_counts().reindex(CALL_TYPES).fillna(0).astype(int)
        reviewed_counts = df_all["call_type"].value_counts().reindex(CALL_TYPES).fillna(0).astype(int)
        comparison = pd.DataFrame({"model label": model_counts, "reviewed label": reviewed_counts})
        st.dataframe(comparison, width="stretch")
        if len(relabeled):
            st.dataframe(
                relabeled[["title", "call_type_model", "call_type", "category"]]
                .rename(columns={"call_type_model": "model label", "call_type": "reviewed label"})
                .sort_values(["reviewed label", "title"]),
                width="stretch",
                hide_index=True,
            )

    topic_counts = pd.Series(
        [topic for topics in df["canonical_topics"] for topic in topics]
    )
    if len(topic_counts):
        st.subheader("Secondary themes")
        show_chart(hbar(topic_counts.value_counts().head(12), "Top themes"))
        st.caption(
            "A theme counts once per meeting. Near-duplicate labels are combined "
            "(“Aegis Detect outage” with “Detect outage”, “Comply v2 launch” with “Comply v2”)."
        )

    st.subheader("One example from each category")
    st.caption(
        "A high-confidence meeting whose title matches the theme. "
        "If no title matches, the highest-confidence meeting in the category is used. "
        "The quote is from the source transcript."
    )
    examples = category_examples(df)
    st.dataframe(
        examples,
        width="stretch",
        hide_index=True,
        column_config={
            "category": st.column_config.TextColumn("Category"),
            "title": st.column_config.TextColumn("Example meeting", width="medium"),
            "call_type": st.column_config.TextColumn("Call type"),
            "sentiment_label": st.column_config.TextColumn("Sentiment"),
            "confidence_score": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "transcript_excerpt": st.column_config.TextColumn("Transcript excerpt", width="large"),
        },
    )

    # ── 2. Sentiment ─────────────────────────────────────────────────────────
    st.header("2. Sentiment across call types")
    st.markdown(
        "Sentiment is the model's label (negative, neutral, positive) and a score from -1 to 1. "
        "The label is the mix. The score is what makes the week-to-week dip visible."
    )

    s1, s2 = st.columns(2)
    with s1:
        show_chart(
            hbar(
                df["sentiment_label"].value_counts().reindex(SENTIMENTS).dropna(),
                "Overall sentiment",
                sort=SENTIMENTS,
            )
        )
    with s2:
        show_chart(stacked_sentiment(df))
    st.markdown(sentiment_note(df))

    r1, r2 = st.columns(2)
    with r1:
        show_chart(stacked_risk(df))
    with r2:
        show_chart(escalation_rate_by_sentiment(df))

    st.markdown(score_conflict_note(df))
    conflicts = df[
        ((df["sentiment_label"] == "negative") & (df["sentiment_score"] > 0))
        | ((df["sentiment_label"] == "positive") & (df["sentiment_score"] < 0))
    ]
    if len(conflicts):
        with st.expander(f"{len(conflicts)} meetings where the label and score disagree"):
            st.dataframe(
                conflicts[
                    ["date", "title", "call_type", "category", "sentiment_label", "sentiment_score"]
                ].sort_values("sentiment_score"),
                width="stretch",
                hide_index=True,
                column_config={"date": st.column_config.DateColumn("Date")},
            )

    st.subheader("Trend")
    show_chart(sentiment_trend(df))
    summary = monthly_summary(df)
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "avg_sentiment_score": st.column_config.NumberColumn("Avg sentiment score", format="%+.2f"),
            "pct_negative": st.column_config.NumberColumn("% negative", format="%.0f"),
            "escalation_risk": st.column_config.NumberColumn("Escalation-risk meetings"),
            "outage_meetings": st.column_config.NumberColumn("Outage meetings"),
        },
    )
    st.markdown(trend_note(df))

    # ── 3. Other insights ────────────────────────────────────────────────────
    st.header("3. Other signals in the transcripts")

    st.subheader("One incident shows up in every call type")
    o1, o2 = st.columns(2)
    with o1:
        outage_types = df.loc[df["mentions_outage"], "call_type"].value_counts().reindex(CALL_TYPES).dropna()
        show_chart(hbar(outage_types, "Outage mentions by call type", sort=CALL_TYPES))
    with o2:
        outage_months = df.loc[df["mentions_outage"], "month"].value_counts().sort_index()
        show_chart(hbar(outage_months, "Outage mentions by month", sort=list(outage_months.index)))
    st.markdown(outage_note(df))

    st.subheader("Which products the calls are about")
    monthly = product_monthly(df)
    show_chart(product_chart(monthly))
    st.markdown(product_note(df, monthly))

    st.subheader("Where escalation risk sits")
    d1, d2 = st.columns(2)
    with d1:
        show_chart(hbar(df.loc[df["escalation_risk"], "category"].value_counts(), "Escalation-risk meetings by category"))
    with d2:
        show_chart(duration_chart(df))
    st.markdown(risk_note(df))

    st.subheader("Escalation-risk meetings")
    risk_df = df[df["escalation_risk"]].copy()
    if risk_df.empty:
        st.success("No escalation-risk meetings in this filter.")
    else:
        label_order = {"negative": 0, "neutral": 1, "positive": 2}
        risk_df["_ord"] = risk_df["sentiment_label"].map(label_order).fillna(3)
        risk_df = risk_df.sort_values(["_ord", "sentiment_score", "date"])
        st.dataframe(
            risk_df[
                [
                    "date",
                    "title",
                    "category",
                    "call_type",
                    "sentiment_label",
                    "sentiment_score",
                    "primary_topic",
                    "llm_summary",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "sentiment_score": st.column_config.NumberColumn("Sentiment score", format="%+.2f"),
                "llm_summary": st.column_config.TextColumn("Summary", width="large"),
            },
        )

    st.divider()
    with st.expander("Filtered rows"):
        show_cols = [
            "date",
            "title",
            "call_type",
            "call_type_model",
            "category",
            "primary_topic",
            "sentiment_label",
            "sentiment_score",
            "escalation_risk",
            "duration_min",
            "confidence_score",
            "llm_summary",
        ]
        st.dataframe(df[show_cols], width="stretch", hide_index=True)
        st.download_button(
            "Download filtered CSV",
            df.drop(columns=["canonical_products", "canonical_topics"]).to_csv(index=False).encode("utf-8"),
            "filtered_results.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
