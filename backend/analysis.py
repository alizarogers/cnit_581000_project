"""
analysis.py: pull structured fields out of PunBot's markdown analysis.

The system prompt asks for a `**Classification**: Pun|Non-pun` line and a
`**Pivot**` section; these helpers read them back so answers can be scored.
"""

import re
from typing import Optional

_PIVOT_HEADING = re.compile(r"^[\s#*_>\-]*(?:\d+[.)]\s*)?[\s*_]*pivot\b", re.IGNORECASE)
_CLASS_RE = re.compile(r"classification\W*?[:\-]\s*\**\s*(non[\s\-]?pun|not a pun|pun)\b", re.IGNORECASE)


def parse_classification(text: str) -> Optional[str]:
    """Return "Pun", "Non-pun", or None if the answer has no classification line."""
    match = _CLASS_RE.search(text or "")
    if not match:
        return None
    return "Pun" if match.group(1).lower() == "pun" else "Non-pun"


def parse_pivot(text: str) -> Optional[str]:
    """Return the pivot word or phrase named in the analysis, if any."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if not _PIVOT_HEADING.match(line):
            continue
        idx = line.lower().find("pivot")
        rest = line[idx + len("pivot"):].lstrip("*_#: \t-")
        if not rest.strip() and i + 1 < len(lines):
            rest = lines[i + 1]
        rest = re.sub(r"[*_`#]", "", rest).strip()
        if re.match(r"(there is no|there's no|no real|no pivot|none\b|n/a)", rest, re.IGNORECASE):
            return None
        rest = re.split(r"\s[(\u2013\-]|[(,;]|\.\s|\bis\b", rest, maxsplit=1)[0]
        rest = rest.strip(" .:\"'\u201c\u201d\u2018\u2019")
        if rest and rest.lower() not in ("none", "n/a", "no pivot"):
            return rest
        return None
    return None


_AGREE_RE = re.compile(r"stage\s*1\s*[-\u2013]?\s*2\s*check\W*?[:\-]\s*\**\s*(partly agree|partially agree|disagree|agree)", re.IGNORECASE)


def parse_stage_check(text: str) -> Optional[str]:
    """Return "Agree", "Partly agree", "Disagree", or None."""
    match = _AGREE_RE.search(text or "")
    if not match:
        return None
    value = match.group(1).lower()
    return "Partly agree" if value.startswith("part") else value.capitalize()
