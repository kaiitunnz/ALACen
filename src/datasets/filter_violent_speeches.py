import json
import requests
import threading
from typing import Callable, Dict, List, Optional
from multiprocessing.pool import ThreadPool
from pathlib import Path


DATA_DIR = Path("violent_speeches")
DATA_PATH = DATA_DIR / "violent_speeches.txt"


class LLMClient:
    def __init__(
        self,
        host: str,
        port: int,
        n: int = 1,
        use_beam_search: bool = False,
        temperature: float = 0,
        max_tokens: int = 10,
        stream: bool = False,
    ):
        self.host = host
        self.port = port
        self.n = n
        self.use_beam_search = use_beam_search
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}/generate"

    def post_http_request(self, prompt: str) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        payload = {
            "prompt": prompt,
            "n": self.n,
            "use_beam_search": self.use_beam_search,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        response = requests.post(
            self.server_url, headers=headers, json=payload, stream=True
        )
        return response

    def get_outputs(self, response: requests.Response) -> List[str]:
        data = json.loads(response.content)
        output = data["text"]
        return output

    def check_violent_speech(self, speech: str) -> Optional[bool]:
        prompt = (
            "Please determine whether the following speech should be censored in movies for children. Please answer either True or False.\n"
            "Speech: What the fuck?\n"
            "Answer: True\n"
            "Speech: God dammit!\n"
            "Answer: True\n"
            "Speech: I love you.\n"
            "Answer: False\n"
            "Speech: I am hungry.\n"
            "Answer: False\n"
            "Speech: you piece of shit\n"
            "Answer: True\n"
            f"Speech: {speech}\n"
            "Answer: "
        )
        response = self.post_http_request(prompt)
        outputs = self.get_outputs(response)
        outputs = self._extract_llm_responses(prompt, outputs)
        violent_count = 0
        nonviolent_count = 0
        unsure_count = 0
        for output in outputs:
            answer = output.lstrip().lower()
            if answer.startswith("true"):
                violent_count += 1
            elif answer.startswith("false"):
                nonviolent_count += 1
            else:
                unsure_count += 1

        if violent_count > nonviolent_count and violent_count > unsure_count:
            return True
        if nonviolent_count > violent_count and nonviolent_count > unsure_count:
            return False
        if unsure_count > violent_count and unsure_count > nonviolent_count:
            return None
        assert False, "Unreachable"

    def _extract_llm_responses(self, prompt: str, outputs: List[str]) -> List[str]:
        return [output[len(prompt) :] for output in outputs]


def init_thread(
    clients: Dict[int, LLMClient], constructor: Callable[[], LLMClient]
) -> None:
    clients[threading.get_ident()] = constructor()


def check_violent_speech(
    clients: Dict[int, LLMClient], speech: str, counter: List[int]
) -> Optional[bool]:
    counter[0] += 1
    if counter[0] % 100 == 0:
        print(f"Processed {counter[0]} speeches", flush=True)
    client = clients[threading.get_ident()]
    return client.check_violent_speech(speech)


def main():
    with open(DATA_PATH) as f:
        speeches = [line.strip() for line in f]

    thread_clients = {}
    counter = [0]
    with ThreadPool(
        8,
        initializer=init_thread,
        initargs=(thread_clients, lambda: LLMClient("143.248.188.103", 8440)),
    ) as pool:
        is_violent_list = pool.starmap(
            check_violent_speech, ((thread_clients, s, counter) for s in speeches)
        )
    violent_speeches = []
    false_positives = []
    uncertain = []
    for speech, is_violent in zip(speeches, is_violent_list):
        if is_violent is None:
            uncertain.append(speech)
        elif is_violent:
            violent_speeches.append(speech)
        else:
            false_positives.append(speech)
    print("Violent speech count:", len(violent_speeches), flush=True)
    print("False positive count:", len(false_positives), flush=True)
    print("Uncertain count:", len(uncertain), flush=True)

    with open(DATA_DIR / "violent_speeches_filtered.txt", "w") as f:
        f.write("\n".join(violent_speeches))
    with open(DATA_DIR / "violent_speeches_removed.txt", "w") as f:
        f.write("\n".join(false_positives))
    with open(DATA_DIR / "violent_speeches_uncertain.txt", "w") as f:
        f.write("\n".join(uncertain))


if __name__ == "__main__":
    main()
