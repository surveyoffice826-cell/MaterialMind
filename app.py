"""MaterialMind - Streamlit app: upload PDF -> agents -> human review -> approve -> download."""
import io
import json
import os
import datetime as dt

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MaterialMind", page_icon="🏗️", layout="wide")

try:
    import crew
    import pdf_extractor
    import report_validator
    from llm import get_api_key
    from stock_calculator import calculate, load_boq
except Exception as exc:  # show a readable message instead of a blank crash
    st.error(f"Setup error while importing modules: {exc}")
    st.stop()

RECORDS, AUDIT = "records.csv", "audit_log.csv"
REC_COLS = ["project", "date", "material", "unit", "quantity", "supplier", "challan_no", "flag"]
TEXT_COLS = ["material", "unit", "supplier", "challan_no", "date", "flag"]


# ---------- helpers ----------
def load_records() -> pd.DataFrame:
    if os.path.exists(RECORDS):
        try:
            df = pd.read_csv(RECORDS, dtype={"project": str, "date": str})
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
            return df.fillna({c: "" for c in REC_COLS if c != "quantity"})
        except Exception:
            pass
    return pd.DataFrame(columns=REC_COLS)


def clean_edited(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in TEXT_COLS:
        d[c] = d[c].fillna("").astype(str).str.strip()
    d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce")
    return d[d["material"] != ""].reset_index(drop=True)


def save_approved(project, date, edited, original, model):
    hist = load_records()
    hist = hist[~((hist["project"] == project) & (hist["date"] == date))]
    new = edited.copy()
    new["project"], new["date"] = project, date
    pd.concat([hist, new[REC_COLS]], ignore_index=True).to_csv(RECORDS, index=False)
    audit = pd.DataFrame([{
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"), "project": project, "date": date,
        "model": model, "human_changed": not clean_edited(original).equals(edited),
        "ai_output": original.to_json(orient="records"), "final_output": edited.to_json(orient="records"),
    }])
    audit.to_csv(AUDIT, mode="a", header=not os.path.exists(AUDIT), index=False)


def to_excel(today_df, result_df, rows_df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        today_df.to_excel(w, sheet_name="Received Today", index=False)
        result_df.to_excel(w, sheet_name="Cumulative & Remaining", index=False)
        rows_df.to_excel(w, sheet_name="Reviewed Rows", index=False)
    return buf.getvalue()


# ---------- sidebar ----------
st.sidebar.title("🏗️ MaterialMind")
project = st.sidebar.text_input("Project name", "Project 1").strip() or "Project 1"
report_date = st.sidebar.date_input("Report date", dt.date.today()).isoformat()
api_in = st.sidebar.text_input("Groq API key", type="password",
                               help="Optional if GROQ_API_KEY is already set in Streamlit Secrets or .env. "
                                    "Kept only in your current browser session.").strip()
if api_in:
    st.session_state["user_api_key"] = api_in
else:
    st.session_state.pop("user_api_key", None)
boq_file = st.sidebar.file_uploader("BOQ (required materials) - CSV/Excel", type=["csv", "xlsx"])
st.sidebar.caption("Columns: material, unit, total_required. Default: sample boq.csv")

try:
    boq = load_boq(boq_file if boq_file else "boq.csv")
except Exception as exc:
    st.error(f"BOQ problem: {exc}")
    st.stop()

st.sidebar.divider()
rec_up = st.sidebar.file_uploader("Restore saved records.csv", type=["csv"])
if rec_up and st.sidebar.button("Restore records"):
    with open(RECORDS, "wb") as f:
        f.write(rec_up.getvalue())
    st.sidebar.success("Records restored")
if os.path.exists(RECORDS):
    st.sidebar.download_button("⬇️ Download records.csv", open(RECORDS, "rb").read(), "records.csv")
st.sidebar.caption("Streamlit Cloud can reset files on restart - download records.csv regularly.")

# ---------- step 1: upload ----------
st.title("MaterialMind")
st.caption("Upload the daily material report. AI agents read it, you review it, then it is saved.")
st.subheader("Step 1 - Upload report")
pdf = st.file_uploader("Daily material report (text PDF)", type=["pdf"])

if st.button("🔍 Process report", type="primary", disabled=pdf is None):
    if not get_api_key():
        st.error("Groq API key is missing. Paste it in the sidebar, or set it in .env / Streamlit Secrets.")
        st.stop()
    text = pdf_extractor.read_pdf(pdf)
    if not text.strip():
        st.error("No text found. Scanned/photo PDFs are not supported - upload a text PDF.")
        st.stop()
    if len(text) > pdf_extractor.MAX_CHARS:
        st.warning(f"Report is long - only the first {pdf_extractor.MAX_CHARS} characters were read. "
                   "Split the PDF if rows are missing.")
    with st.spinner("Agents are reading the report (may take up to a minute)..."):
        try:
            df, model = crew.extract_and_normalize(text, boq)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    st.session_state.update(ai_df=df, orig_df=df.copy(), model=model, validation=None,
                            saved=False, run_id=st.session_state.get("run_id", 0) + 1)

# ---------- step 2: human review ----------
if "ai_df" in st.session_state:
    st.subheader("Step 2 - Human review (edit anything that is wrong)")
    st.caption(f"Model used: {st.session_state.model}. Rows flagged 'Not in BOQ' / 'Check unit' need attention.")
    edited_raw = st.data_editor(
        st.session_state.ai_df, num_rows="dynamic", use_container_width=True,
        key=f"editor_{st.session_state.run_id}",
        column_config={"quantity": st.column_config.NumberColumn("quantity", format="%.3f")},
    )
    edited = clean_edited(edited_raw)
    result = calculate(edited, load_records(), boq, report_date, project)
    issues = report_validator.python_checks(edited, result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(edited))
    c2.metric("Materials received today", int((result["received_today"] > 0).sum()))
    c3.metric("Issues found", len(issues))

    tab1, tab2, tab3 = st.tabs(["Received today", "Total received", "Remaining needed"])
    today_df = result[result["received_today"] > 0][["material", "unit", "received_today"]]
    tab1.dataframe(today_df, use_container_width=True, hide_index=True)
    tab2.dataframe(result[result["total_received"] > 0][["material", "unit", "total_received", "total_required"]],
                   use_container_width=True, hide_index=True)
    tab3.dataframe(result[["material", "unit", "total_required", "total_received", "remaining", "status"]],
                   use_container_width=True, hide_index=True)

    st.subheader("Step 3 - Validation")
    for item in issues:
        st.warning(item)
    if not issues:
        st.success("No rule-based issues found.")
    if st.button("🤖 Run AI validator agent"):
        with st.spinner("Validator agent is checking..."):
            try:
                st.session_state.validation, _ = crew.run_validator(result, issues)
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("validation"):
        st.info(st.session_state.validation)

    st.subheader("Step 4 - Approve")
    ok = st.checkbox("I have checked and corrected all values")
    a, b = st.columns(2)
    if a.button("✅ Approve & Save", type="primary", disabled=not ok):
        save_approved(project, report_date, edited, st.session_state.orig_df, st.session_state.model)
        st.session_state.saved = True
        st.success(f"Saved for {project} on {report_date}.")
    if b.button("❌ Reject"):
        for k in ("ai_df", "orig_df", "validation"):
            st.session_state.pop(k, None)
        st.rerun()

    st.download_button("⬇️ Download Excel report", to_excel(today_df, result, edited),
                       f"materialmind_{project}_{report_date}.xlsx")
