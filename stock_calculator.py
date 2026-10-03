"""Agent 3 - Stock Calculator. Pure Python/pandas math (the LLM never does arithmetic)."""
import pandas as pd

# Known unit conversions (from, to) -> factor. Anything else is flagged "Unit mismatch".
FACTORS = {
    ("kg", "ton"): 0.001, ("ton", "kg"): 1000.0,
    ("m3", "cft"): 35.3147, ("cft", "m3"): 0.0283168,
    ("ltr", "m3"): 0.001, ("m3", "ltr"): 1000.0,
    ("rft", "mtr"): 0.3048, ("mtr", "rft"): 3.28084,
}
RESULT_COLS = ["material", "unit", "received_today", "total_received", "total_required", "remaining", "status"]


def _key(value) -> str:
    return str(value).strip().lower()


def load_boq(source) -> pd.DataFrame:
    """Read BOQ (csv/xlsx path or uploaded file) with columns: material, unit, total_required."""
    name = getattr(source, "name", str(source)).lower()
    df = pd.read_excel(source) if name.endswith((".xlsx", ".xls")) else pd.read_csv(source)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = {"material", "unit", "total_required"} - set(df.columns)
    if missing:
        raise ValueError(f"BOQ is missing columns: {', '.join(sorted(missing))}")
    df = df[["material", "unit", "total_required"]].dropna(subset=["material"]).copy()
    df["material"] = df["material"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["total_required"] = pd.to_numeric(df["total_required"], errors="coerce")
    return df.reset_index(drop=True)


def align_units(today: pd.DataFrame, boq: pd.DataFrame) -> pd.DataFrame:
    """Convert rows to the BOQ unit when a known factor exists, else flag them."""
    df = today.copy()
    boq_unit = {_key(m): u for m, u in zip(boq["material"], boq["unit"])}
    for i, r in df.iterrows():
        target = boq_unit.get(_key(r["material"]))
        if target is None or pd.isna(r["quantity"]) or _key(r["unit"]) == _key(target):
            continue
        factor = FACTORS.get((_key(r["unit"]), _key(target)))
        if factor:
            df.at[i, "quantity"] = r["quantity"] * factor
            df.at[i, "unit"] = target
            df.at[i, "flag"] = (str(r["flag"]) + " Converted").strip()
        else:
            df.at[i, "flag"] = (str(r["flag"]) + " Unit mismatch").strip()
    return df


def calculate(today: pd.DataFrame, history: pd.DataFrame, boq: pd.DataFrame,
              report_date: str, project: str) -> pd.DataFrame:
    """Return per-material: received today, total received, required, remaining, status."""
    today = align_units(today, boq)
    if not history.empty:
        history = history[(history["project"] == project) & (history["date"] != report_date)]

    slots = {}

    def slot(material, unit):
        return slots.setdefault((_key(material), _key(unit)), {
            "material": material, "unit": unit, "today": 0.0, "previous": 0.0, "required": None})

    for _, r in boq.iterrows():
        slot(r["material"], r["unit"])["required"] = r["total_required"]
    for _, r in history.iterrows():
        if pd.notna(r["quantity"]):
            slot(r["material"], r["unit"])["previous"] += float(r["quantity"])
    for _, r in today.iterrows():
        if pd.notna(r["quantity"]):
            slot(r["material"], r["unit"])["today"] += float(r["quantity"])

    out = []
    for v in slots.values():
        total = v["previous"] + v["today"]
        required = v["required"] if pd.notna(v["required"]) else None
        remaining = None if required is None else required - total
        if required is None:
            status = "Not in BOQ"
        elif remaining < 0:
            status = "Over-received"
        elif remaining == 0:
            status = "Complete"
        else:
            status = "Pending"
        out.append({"material": v["material"], "unit": v["unit"],
                    "received_today": round(v["today"], 3), "total_received": round(total, 3),
                    "total_required": required, "remaining": None if remaining is None else round(remaining, 3),
                    "status": status})
    return pd.DataFrame(out, columns=RESULT_COLS)
