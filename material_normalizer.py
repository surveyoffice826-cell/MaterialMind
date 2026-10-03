"""Agent 2 - Material Normalizer: standard names and units (LLM + Python clean-up)."""
import difflib

import pandas as pd
from crewai import Agent, Task

UNIT_MAP = {
    "bag": "bags", "bags": "bags", "ton": "ton", "tons": "ton", "tonne": "ton", "tonnes": "ton", "mt": "ton",
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "cft": "cft", "cubic feet": "cft", "cu ft": "cft",
    "m3": "m3", "m³": "m3", "cum": "m3", "cubic meter": "m3", "no": "nos", "nos": "nos", "pcs": "nos",
    "pc": "nos", "piece": "nos", "pieces": "nos", "ltr": "ltr", "liter": "ltr", "litre": "ltr", "l": "ltr",
    "rft": "rft", "running feet": "rft", "mtr": "mtr", "meter": "mtr", "m": "mtr", "sqft": "sqft",
    "sq ft": "sqft", "set": "set", "sets": "set", "roll": "roll", "rolls": "roll",
}


def clean_unit(unit) -> str:
    key = str(unit or "").strip().lower().replace(".", "")
    return UNIT_MAP.get(key, key)


def build(llm, boq_names, context_task):
    agent = Agent(
        role="Material Name and Unit Normalizer",
        goal="Make material names and units consistent with the project BOQ without changing any quantity.",
        backstory="You are a quantity surveyor who knows that 'OPC 50kg' and 'Cement bag' are the same item.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    task = Task(
        description=(
            "Take the extracted rows and normalize them.\n"
            "BOQ materials: " + boq_names + "\n"
            "Rules: if a row is the same material as a BOQ item, use the exact BOQ name. "
            "Use short lowercase units (bags, ton, kg, cft, m3, nos, rft, mtr, sqft, ltr, set, roll). "
            "Never change quantities, never add or remove rows. "
            "If no BOQ item matches, keep the name and set flag to Not in BOQ. "
            "If the unit is unclear, set flag to Check unit.\n"
            "Return ONLY a JSON array with keys: material, unit, quantity, supplier, challan_no, date, flag."
        ),
        expected_output="A JSON array of normalized material rows.",
        agent=agent, context=[context_task],
    )
    return agent, task


def post_clean(rows, boq) -> pd.DataFrame:
    """Python safety net: exact/fuzzy BOQ name match, unit clean-up, review flags."""
    names = {str(m).strip().lower(): str(m).strip() for m in boq["material"]}
    for r in rows:
        r.unit = clean_unit(r.unit)
        key = r.material.strip().lower()
        if key in names:
            r.material = names[key]
        else:
            close = difflib.get_close_matches(key, list(names), n=1, cutoff=0.82)
            if close:
                r.material = names[close[0]]
            elif "Not in BOQ" not in r.flag:
                r.flag = (r.flag + " Not in BOQ").strip()
        if not r.unit and "Check unit" not in r.flag:
            r.flag = (r.flag + " Check unit").strip()
        if r.quantity is None and "Check quantity" not in r.flag:
            r.flag = (r.flag + " Check quantity").strip()
    return pd.DataFrame([r.model_dump() for r in rows])
