"""Agent 1 - PDF Extraction Specialist: reads the report text and returns structured rows."""
import pdfplumber
from crewai import Agent, Task

MAX_CHARS = 8000  # keeps each request inside the Groq free-tier token limit


def read_pdf(file) -> str:
    """Extract text from a text-based PDF (tables are flattened to 'a | b | c' lines)."""
    lines = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        cells = [str(c).replace("\n", " ").strip() for c in row if c]
                        if cells:
                            lines.append(" | ".join(cells))
            else:
                lines.append(page.extract_text() or "")
    # braces would be treated as template variables by CrewAI
    return "\n".join(lines).replace("{", "(").replace("}", ")")


def build(llm, text):
    agent = Agent(
        role="PDF Data Extraction Specialist",
        goal="Extract every received-material line from a daily site material report, exactly as written.",
        backstory="You are a careful construction store-keeper who copies quantities exactly and never guesses.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    task = Task(
        description=(
            "Below is the text of a daily material report. Extract every material RECEIVED line.\n"
            "Return ONLY a JSON array. Each item has the keys: material, unit, quantity, supplier, challan_no, date.\n"
            "Rules: copy numbers exactly, use null when a value is missing, never invent rows, no explanation text.\n\n"
            "REPORT TEXT:\n" + text[:MAX_CHARS]
        ),
        expected_output="A JSON array of material rows.",
        agent=agent,
    )
    return agent, task
