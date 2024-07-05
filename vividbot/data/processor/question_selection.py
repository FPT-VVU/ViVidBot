import json
import random

from vividbot.data.processor.base import BaseProcessor

# set seed
random.seed(42)


class QuestionSelection(BaseProcessor):
    """
    This class and its children are used to generate prompts.
    """

    def __init__(self, question_path: str, tags: str = "<video>", **kwargs):
        # read file
        self.question_list = []
        with open(question_path, "r") as f:
            for line in f:
                self.question_list.append(line.strip() + "\n" + tags)

    def process(self, num_gen: int = 1, *args, **kwargs) -> dict:
        return [random.choice(self.question_list) for _ in range(num_gen)]
