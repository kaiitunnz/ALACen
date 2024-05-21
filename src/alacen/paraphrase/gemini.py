from typing import List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.ai.generativelanguage import Candidate

from . import ParaphraseGenerator
from .. import config


class GeminiClient(ParaphraseGenerator):
    def __init__(self):
        genai.configure(api_key=config.api_key)
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

    def paraphrase(self, speech: str, n: int = 1) -> List[str]:
        prompt = self.get_paraphrase_prompt(speech)
        candidates: List[Candidate] = []
        for _ in range(n):
            outputs = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            candidates.extend(outputs.candidates)
        scores = self.score_outputs(speech, candidates)
        candidate_strs = [
            " ".join(part.text for part in candidate.content.parts)
            for candidate in candidates
        ]
        output = sorted(zip(scores, candidate_strs))
        return [o[-1] for o in output]

    def score_outputs(self, speech: str, outputs: List[Candidate]) -> List[float]:
        speech_tokens = self.model.count_tokens(speech).total_tokens
        return [abs(speech_tokens - output.token_count) for output in outputs]
