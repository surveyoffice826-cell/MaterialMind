"""Strict data models + safe JSON parsing for agent output."""
import json
import re
from typing import List, Optional

from pydantic import BaseModel, field_validator


class MaterialRow(BaseModel):
    material: str
    unit: str = ""
    quantity: Optional[float] = None
    supplier: str = ""
    challan_no: str = ""
    date: str = ""
    flag: str = ""

    @field_validator("quantity", mode="before")
    @classmethod
    def _qty(cls, v):
        if v in (None, ""):
            return None
        try:
            return float(str(v).replace(",", "").strip())
        except ValueError:
            return None

    @field_validator("material", "unit", "supplier", "challan_no", "date", "flag", mode="before")
    @classmethod
    def _text(cls, v):
        return "" if v is None else str(v).strip()


def extract_json(text):
    """Find the first valid JSON array/object inside an LLM answer."""
    text = re.sub(r"```(?:json)?", "", str(text))
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_rows(text) -> List[MaterialRow]:
    """Turn agent output into validated MaterialRow objects (bad rows are skipped)."""
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get("rows", [])
    rows = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(MaterialRow(**{k: v for k, v in item.items() if k in MaterialRow.model_fields}))
        except Exception:
            continue
    return rows
