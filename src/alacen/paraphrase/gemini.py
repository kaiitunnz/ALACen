import os
from typing import List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.ai.generativelanguage import Candidate

from . import ParaphraseGenerator
from .utils import count_syllables
from .. import config


class GeminiClient(ParaphraseGenerator):
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", None)
        genai.configure(api_key=api_key or config.api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        self.generation_config = genai.GenerationConfig(
            candidate_count=config.candidate_count,
            temperature=config.temperature,
        )
        self.safety_settings = {
            # HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
        }

    def get_paraphrase_prompt(self, speech: str) -> str:
        return (
            "Please paraphrase the following speech from violent to non-violent. "
            "Keep the length of the resulting speech similar to the original speech. "
            "Keep the tone and naturalness of the speech similar to the original speech.\n"
            "Speech: What the fuck?\n"
            "Answer: Oh my god!\n"
            f"Speech: {speech}\n"
            "Answer: "
        )

    def paraphrase(self, speech: str, n: int = 1, **_) -> List[str]:
        prompt = self.get_paraphrase_prompt(speech)
        candidates: List[Candidate] = []
        for _ in range(n):
            outputs = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            candidates.extend(outputs.candidates)

        candidate_strs = [
            " ".join(part.text for part in candidate.content.parts)
            for candidate in candidates
        ]
        speech_syllables = count_syllables(speech)
        output = sorted(
            candidate_strs, key=lambda x: abs(speech_syllables - count_syllables(x))
        )
        return output
