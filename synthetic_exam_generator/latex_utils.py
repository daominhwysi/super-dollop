import re
from typing import List, Tuple


def is_valid_latex(content: str) -> bool:
    """Check if content string represents valid LaTeX math notation."""
    content_stripped = content.strip()
    if not content_stripped:
        return False

    # Case 1: Single variable or number (e.g. $x$, $a$, $1$)
    if len(content_stripped) == 1:
        return content_stripped.isalnum()

    # Case 2: Mathematical interval notation like (-\infty; 0] or [8; +\infty) or (1; 2]
    if re.match(r'^[\[\(][^\[\]\(\)]+[\]\)]$', content_stripped) and (';' in content_stripped or ',' in content_stripped):
        return True

    # Case 3: Balanced braces, brackets, and parentheses
    brackets = {'{': '}', '(': ')', '[': ']'}
    stack = []
    for char in content_stripped:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False
            last = stack.pop()
            if brackets[last] != char:
                return False
    if stack:
        return False

    # Case 4: Contains standard math/latex character indicators
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True

    # Case 5: Short alphanumeric math terms without spaces (e.g. $2a$, $x1$, $100$)
    if len(content_stripped) < 10 and re.match(r'^[a-zA-Z0-9]+$', content_stripped):
        return True

    return False


def get_latex_spans(text: str) -> List[Tuple[int, int]]:
    """Return character start and end spans of valid LaTeX formulas in text."""
    spans = []
    # 1. Matches $$...$$ (display math)
    for m in re.finditer(r'\$\$(.+?)\$\$', text, flags=re.DOTALL):
        if is_valid_latex(m.group(1)):
            spans.append((m.start(), m.end()))
    # 2. Matches $...$ (inline math, excluding $$)
    for m in re.finditer(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', text):
        if is_valid_latex(m.group(1)):
            spans.append((m.start(), m.end()))
    return spans
