from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("d3po_pytorch.assets")
index = 0


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `d3po_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def from_file_all(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return prompts


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")

def anything_prompt():
    return from_file("anything_prompt.txt")

def unsafe_prompt():
    return from_file("unsafe_prompt.txt")

def complex_animals():
    return from_file("complex_animals.txt")

def simple_prompt():
    global index  # 允许修改全局变量 index
    prompts = from_file_all("simple_prompt.txt")
    if index >= len(prompts):  # 如果索引超出范围，重置为 0
        index = 0
    prompt = prompts[index]
    index += 1  # 更新索引，下一次调用返回下一个 prompt
    return prompt, {}

# if __name__ == "__main__":
#     print(simple_prompt())
#     print(simple_prompt())
#     print(simple_prompt())
#     print(simple_prompt())
