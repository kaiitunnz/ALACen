import json
import threading
import time
from typing import Dict, List
from multiprocessing.pool import ThreadPool
from pathlib import Path

import atomics  # type: ignore
import google.generativeai as genai  # type: ignore
from atomics._impl.atomic.int import AtomicInt  # type: ignore
from google.ai.generativelanguage import Candidate  # type: ignore
from google.api_core.retry import Retry  # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore

import config

DATA_DIR = Path("violent_speeches")
DATA_PATH = DATA_DIR / "violent_speech_final.txt"
RESULT_PATH = DATA_DIR / "violent_speech_dataset.json"

API_KEYS: List[str] = []  # Put your API keys here

BATCH_SIZE = 10


class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        self.num_trials = config.num_trials
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
            "Keep the tone and naturalness of the speech similar to the original speech. "
            "Make sure that there is no impolite words."
            "Reply only the paraphrased speech and nothing else.\n"
            f"Speech: {speech}\n"
            "Paraphrased: "
        )

    def paraphrase_violent_speech(self, speech: str) -> str:
        prompt = self.get_paraphrase_prompt(speech)
        candidates: List[Candidate] = []
        for _ in range(self.num_trials):
            candidates.extend(self.send_request(prompt))
        scores = self.score_outputs(speech, candidates)
        candidate_strs = [
            (" ".join(part.text for part in candidate.content.parts)).rstrip()
            for candidate in candidates
        ]
        try:
            output = sorted(zip(scores, candidate_strs))[-1][1]
        except IndexError:
            output = "<ERROR>"
        return output

    def send_request(self, prompt: str) -> List[Candidate]:
        outputs = self.model.generate_content(
            prompt,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            request_options={"retry": Retry(timeout=None)},
        )
        return outputs.candidates

    def score_outputs(self, speech: str, outputs: List[Candidate]) -> List[float]:
        speech_tokens = self.model.count_tokens(speech).total_tokens
        return [-abs(speech_tokens - output.token_count) for output in outputs]


def init_thread(clients: Dict[int, GeminiClient]) -> None:
    tid = threading.get_ident()
    if tid not in clients:
        clients[tid] = GeminiClient(API_KEYS.pop())


def paraphrase_violent_speech(
    clients: Dict[int, GeminiClient],
    speech: str,
    counter: AtomicInt,
    total: int,
    start_time: List[float],
) -> str:
    client = clients[threading.get_ident()]
    paraphrased = client.paraphrase_violent_speech(speech)
    i = counter.fetch_inc()
    if (i % BATCH_SIZE) == 0:
        new_time = time.time()
        print(
            f"Processed {i}/{total} speeches (time taken: {new_time - start_time[0]} seconds)",
            flush=True,
        )
        start_time[0] = new_time
    return paraphrased


def main():
    with open(DATA_PATH) as f:
        speeches = [line.strip() for line in f]

    if RESULT_PATH.exists():
        with open(RESULT_PATH) as f:
            start = len(f.readlines())
    else:
        start = 0

    thread_clients = {}
    counter = atomics.atomic(4, atype=atomics.INT)
    counter.store(start + 1)
    start_time = [time.time()]

    with open(RESULT_PATH, "a") as f:
        with ThreadPool(
            len(API_KEYS),
            initializer=init_thread,
            initargs=(thread_clients,),
        ) as pool:
            for i in range(start, len(speeches), BATCH_SIZE):
                batch = speeches[i : i + BATCH_SIZE]
                paraphrased_list = pool.starmap(
                    paraphrase_violent_speech,
                    (
                        (thread_clients, s, counter, len(speeches), start_time)
                        for s in batch
                    ),
                )
                for speech, paraphrased in zip(batch, paraphrased_list):
                    json.dump({"text": speech, "paraphrase": paraphrased}, f)
                    f.write("\n")
                    f.flush()

    with open(RESULT_PATH) as f:
        dataset = [json.loads(line.strip()) for line in f]
    with open(RESULT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    main()
