import re

import cmudict
import syllables

cmu_dict = cmudict.dict()


def count_syllables_in_word(word: str) -> int:
    phones = cmu_dict.get(word.lower())
    if phones:
        return len([p for p in phones[0] if p[-1].isdigit()])
    return syllables.estimate(word)


def count_syllables(text: str) -> int:
    words = re.findall(r"\w+", text)
    return sum(count_syllables_in_word(word) for word in words)
