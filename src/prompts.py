from typing import List, Dict, Tuple

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers strictly using the provided context. "
    "Cite inline like [Title §Section, Year]. If information is missing, say you don't know."
)

def build_prompt(user_query: str, retrieved: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Returns (system_prompt, context_list_for_display)
    We keep this simple—production code would create a structured prompt.
    """
    summaries = []
    for r in retrieved:
        # Keeping a tiny summary so the demo doesn't require an LLM just to condense
        preview = (r["text"].split("\n")[0])[:220]
        summaries.append({
            "inline_citation": r["inline_citation"],
            "summary": preview,
            "score": r["score"],
            "title": r["title"],
            "source": r["source"],
            "url": r["url"],
            "section": r.get("section",""),
            "year": r.get("year",""),
            "preview": r["preview"]
        })

    system_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUser question: {user_query}\n"
    return system_prompt, summaries
