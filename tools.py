"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    listings = load_listings()

    # Filter by price and size
    if max_price is not None:
        listings = [l for l in listings if l["price"] <= max_price]
    if size is not None:
        size_lower = size.lower()
        listings = [l for l in listings if size_lower in l["size"].lower()]

    # Score by keyword overlap with description
    keywords = set(description.lower().split())

    def score(listing):
        # Build a string called text from all the relevant fields of a listing
        text = " ".join([
            listing["title"],
            listing["description"],
            listing["category"],
            " ".join(listing.get("style_tags", [])),
            " ".join(listing.get("colors", [])),
            listing.get("brand") or "",
        ]).lower()
        # Count up the number of keywords from the description that matches the listing string
        return sum(1 for kw in keywords if kw in text)

    scored = [(score(l), l) for l in listings] # Builds a list of (score, listing) tuples for every listing
    scored = [(s, l) for s, l in scored if s > 0] # Drops any listing where no keywords matched (score = 0)
    scored.sort(key=lambda x: x[0], reverse=True) # Sort by descending order
    return [l for _, l in scored] # Strip the scores out and returns the listing dicts in ranked order


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    # Guard against missing or empty new_item
    if not new_item:
        return "Could not generate an outfit suggestion: no item was provided."

    client = _get_groq_client()

    # Build a readable description of the new item to include in the prompt
    item_desc = (
        f"{new_item['title']} — {new_item['description']} "
        f"(category: {new_item.get('category', 'unknown')}; "
        f"brand: {new_item.get('brand') or 'unbranded'}; "
        f"colors: {', '.join(new_item.get('colors', []))}; "
        f"style: {', '.join(new_item.get('style_tags', []))})"
    )

    wardrobe_items = wardrobe.get("items", [])

    if not wardrobe_items:
        # Wardrobe is empty, ask for general styling advice instead
        prompt = (
            f"I'm considering buying this thrifted item: {item_desc}\n\n"
            "My wardrobe is currently empty. Suggest one complete outfit idea that includes "
            "a top, bottom, shoes, outerwear, and accessory that would pair well with it. "
            "Describe the vibe it suits. Be specific and keep it to 3–5 sentences."
        )
    else:
        # Format each wardrobe item as a bullet for the prompt
        wardrobe_text = "\n".join(
            f"- {item['name']} ({item['category']}, colors: {', '.join(item.get('colors', []))}, "
            f"style: {', '.join(item.get('style_tags', []))}"
            + (f", notes: {item['notes']}" if item.get('notes') else "") + ")"
            for item in wardrobe_items
        )
        # Ask the LLM to suggest one outfit using specific named wardrobe pieces
        prompt = (
            f"I'm considering buying this thrifted item: {item_desc}\n\n"
            f"Here are the pieces already in my wardrobe:\n{wardrobe_text}\n\n"
            "Suggest one complete outfit using the new item paired with specific pieces "
            "from my wardrobe. Try to include as many categories as possible (top, bottom, "
            "shoes, outerwear, accessory) using only pieces from my wardrobe. "
            "Name each piece and explain why the combination works. Keep it to 3–5 sentences."
        )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # Guard against empty or whitespace-only outfit string
    if not outfit or not outfit.strip():
        return "Could not generate a fit card: no outfit suggestion was provided."

    client = _get_groq_client()

    prompt = (
        f"Here is a thrifted item: {new_item['title']} — found on {new_item['platform']} for ${new_item['price']}.\n\n"
        f"Here is the suggested outfit: {outfit}\n\n"
        "Write a 2–4 sentence Instagram caption for this outfit. Use a casual, authentic OOTD tone with emojis. "
        "Naturally mention the item name, price, and platform once each. Capture the outfit vibe in specific terms."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return response.choices[0].message.content
