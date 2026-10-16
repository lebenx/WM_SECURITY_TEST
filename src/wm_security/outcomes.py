"""Conservative coding of explicit ScienceWorld move feedback.

Descriptions of a whole room remain unknown: this parser is not a semantic
judge. Only the leading outcome clause is used, so an unrelated preservation
statement such as 'moving does not change its state of matter' cannot turn an
explicitly successful move into a failure.
"""
import re


MOVE_CODING_VERSION = 3


def move_outcome(text):
    s = re.sub(r'^\s*prediction:\s*', '', text.lower()).strip()
    leading = re.split(r'(?<=[.!?])\s+', s, maxsplit=1)[0]
    if re.search(r"\b(can't|cannot|unable|fails?|not possible)\b", leading):
        return False
    if re.search(r'\b(?:need|must) to open\b', leading):
        return False
    if re.search(r'\b(?:try|attempt) to\b', leading):
        return False if re.search(r'\bbut\b.*\bclosed\b', leading) else None
    if re.match(r'you (?:successfully )?(?:move|place|put|transfer)\b', leading):
        if re.search(r'\bbut\b', leading):
            return None
        return True
    return None
