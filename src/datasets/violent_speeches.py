import os
import threading
import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from transformers import Pipeline, pipeline  # type: ignore


BATCH_SIZE = 512
RESULT_DIR = Path("violent_speeches")
DATA_DIR = Path("scripts")

devices = [0, 1]


class Script:
    def __init__(self, data_dir: Path, movie_id: str, concat: bool = True):
        self.data_dir = data_dir
        self.movie_id = movie_id
        self.concat = concat
        self.lines = self.load_script()

    def load_script(self) -> List[Tuple[str, str]]:
        fdir = self.data_dir / self.movie_id
        result: List[Tuple[str, str]] = []
        with open(fdir / "script.txt") as script, open(
            fdir / "rule-parse.txt"
        ) as labels:
            if len(script.readlines()) != len(labels.readlines()):
                print(
                    "warning: inconsistent script and label files for movie:",
                    self.movie_id,
                )
            script.seek(0)
            labels.seek(0)
            for line, label in zip(script, labels):
                result.append((label.strip(), line.strip()))
        return result

    def _get_lines(self, label: Optional[str] = None):
        if not self.concat:
            return [line for lab, line in self.lines if lab == label or label is None]
        result: List[str] = []
        prev_lab = None
        for lab, line in self.lines:
            if lab == label or label is None:
                if lab == prev_lab:
                    result[-1] += " " + line
                else:
                    result.append(line)
            prev_lab = lab
        return result

    @property
    def sluglines(self):
        return self._get_lines("S")

    @property
    def descriptions(self):
        return self._get_lines("N")

    @property
    def characters(self):
        return self._get_lines("C")

    @property
    def utterances(self):
        return self._get_lines("D")

    @property
    def utterance_expressions(self):
        return self._get_lines("E")

    @property
    def transitions(self):
        return self._get_lines("T")

    @property
    def metadata(self):
        return self._get_lines("M")

    @property
    def others(self):
        return self._get_lines("O")


def init_thread(classifiers: Dict[int, Pipeline]):
    classifiers[threading.get_ident()] = pipeline(
        "text-classification",
        model="uhhlt/bert-based-uncased-hatespeech-movies",
        device=devices.pop(),
    )


def get_violent_utterances(
    classifiers: Dict[int, Pipeline],
    data_dir: Path,
    movie_id: str,
    stat_dict: Dict[str, List],
):
    start_time = time.time()

    classifier = classifiers[threading.get_ident()]
    script = Script(data_dir, movie_id)

    result: List[str] = []
    utterances = script.utterances
    for j in range(0, len(utterances), BATCH_SIZE):
        inputs = utterances[j : j + BATCH_SIZE]
        preds = classifier(inputs)
        result.extend(
            utter for utter, pred in zip(inputs, preds) if pred["label"] != "Normal"
        )

    stat_dict["utterance_counts"].append(len(script.utterances))
    stat_dict["violent_utterance_counts"].append(len(result))
    # Write to a result file.
    with open(RESULT_DIR / (movie_id + ".txt"), "w") as f:
        f.write("\n".join(result))

    print(
        f"DONE: movie_id={movie_id}, time_elapsed={time.time() - start_time:.4f}",
        flush=True,
    )


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    classifiers = {}
    stat_dict = defaultdict(list)

    with ThreadPool(len(devices), init_thread, (classifiers,)) as pool:
        pool.starmap(
            get_violent_utterances,
            (
                (classifiers, DATA_DIR, movie_id, stat_dict)
                for movie_id in os.listdir(DATA_DIR)
            ),
        )

    stat_df = pd.DataFrame(stat_dict)
    print(stat_df.describe(), flush=True)
    print(
        "Total number of violent utterances:",
        stat_df["violent_utterance_counts"].sum(),
        flush=True,
    )


if __name__ == "__main__":
    main()
