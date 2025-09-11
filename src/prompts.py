# src/prompts.py
def build_system_prompt(org_name: str) -> str:
    return (
        f"You are a retrieval assistant for {org_name}. "
        "Answer in 1–3 short sentences. Only use information in the retrieved passages. "
        "If uncertain, say you don’t know and suggest which document to check."
    )

