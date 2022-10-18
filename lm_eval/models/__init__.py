from . import gpt2
from . import gpt3
from . import dummy
from . import glm_130b

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "dummy": dummy.DummyLM,
    "glm_130b": glm_130b.GLM130BLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
