import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Digital channel feature
    if "Submitted via" in df.columns:
        df["Digital_Channel"] = df["Submitted via"].isin(["Web"]).astype(int)
    else:
        df["Digital_Channel"] = 0

    # Date-based features
    date_received_col = "Date received"
    date_sent_col = "Date sent to company"

    if date_received_col in df.columns:
        dr = pd.to_datetime(df[date_received_col], errors="coerce")
        df["received_dow"] = dr.dt.dayofweek
        df["received_month"] = dr.dt.month
    else:
        df["received_dow"] = pd.NA
        df["received_month"] = pd.NA

    if date_received_col in df.columns and date_sent_col in df.columns:
        dr = pd.to_datetime(df[date_received_col], errors="coerce")
        ds = pd.to_datetime(df[date_sent_col], errors="coerce")
        df["days_to_company"] = (ds - dr).dt.days
    else:
        df["days_to_company"] = pd.NA

    # Narrative light features (no NLP)
    narrative_col = "Consumer complaint narrative"
    if narrative_col in df.columns:
        txt = df[narrative_col].fillna("").astype(str)
        df["has_narrative"] = (txt.str.len() > 0).astype(int)
        df["narrative_len"] = txt.str.len()
    else:
        df["has_narrative"] = 0
        df["narrative_len"] = 0

    # Missingness flags
    if "Tags" in df.columns:
        df["has_tags"] = df["Tags"].notna().astype(int)
    else:
        df["has_tags"] = 0

    if "Sub-issue" in df.columns:
        df["has_sub_issue"] = df["Sub-issue"].notna().astype(int)
    else:
        df["has_sub_issue"] = 0

    # Target 
    if "Consumer disputed?" in df.columns:
        df["Target"] = df["Consumer disputed?"].map({"Yes": 1, "No": 0})

    return df