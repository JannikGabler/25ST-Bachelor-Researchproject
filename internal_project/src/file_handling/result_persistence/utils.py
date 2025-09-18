from __future__ import annotations

def slugify(text: str) -> str:
    """
    Convert an arbitrary string into a filesystem-safe slug.

    - Keeps only alphanumeric characters, dashes (`-`), and underscores (`_`).
    - Replaces all other characters with underscores.
    - Collapses consecutive underscores into a single underscore.
    - Strips leading/trailing whitespace before processing.
    - Returns "artifact" if the result would otherwise be empty.
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "artifact"
