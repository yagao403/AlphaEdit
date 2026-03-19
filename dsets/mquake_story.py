import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *



class MQUAKE_STORY_Dataset:


    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_t_loc = data_dir / "mquake_story_question_data.json"
        if not mquake_t_loc.exists():
            print(f"{mquake_t_loc} does not exist.")
            raise FileNotFoundError(f"{mquake_t_loc} does not exist.")

        with open(mquake_t_loc, "r") as f:
            data = json.load(f)


        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
