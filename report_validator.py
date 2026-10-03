"""Agent 4 - Report Validator: Python checks + an LLM review of the final table."""
import pandas as pd
from crewai import Agent, Task


def python_checks(df: pd.DataFrame, result: pd.DataFrame) -> list:
    """Fast rule-based checks shown to the human reviewer."""
    issues = []
    for n, (_, r) in enumerate(df.iterrows(), start=1):
        if not str(r["material"]).strip():
            issues.append(f"Row {n}: material name missing")
        if pd.isna(r["quantity"]) or r["quantity"] <= 0:
            issues.append(f"Row {n} ({r['material']}): quantity missing or not positive")
        if not str(r["unit"]).strip():
            issues.append(f"Row {n} ({r['material']}): unit missing")
        for tag in ("Not in BOQ", "Unit mismatch"):
            if tag in str(r["flag"]):
                issues.append(f"Row {n} ({r['material']}): {tag}")
    challan = df["challan_no"].astype(str).str.strip() != ""
    dup = df[challan & df.duplicated(["material", "challan_no", "quantity"], keep=False)]
    if not dup.empty:
        issues.append("Possible duplicate entries (same material, challan and quantity): "
                      + ", ".join(sorted(set(dup["material"].astype(str)))))
    for _, r in result.iterrows():
        if r["status"] == "Over-received":
            issues.append(f"{r['material']}: received more than required")
    return issues


def build(llm, table_csv, issues):
    agent = Agent(
        role="Site Material Quality Checker",
        goal="Review the final material table and point out anything a human should double-check.",
        backstory="You are a strict senior store auditor on a construction project.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    task = Task(
        description=(
            "Review this stock table (CSV) and the issues already found by code.\n"
            "List at most 5 short bullet points: suspicious quantities, unit problems, missing data. "
            "If everything looks fine, answer only: OK.\n\n"
            "ISSUES FOUND: " + ("; ".join(issues) if issues else "none") + "\n\n"
            "TABLE:\n" + table_csv.replace("{", "(").replace("}", ")")
        ),
        expected_output="Up to 5 short bullet points, or OK.",
        agent=agent,
    )
    return agent, task
