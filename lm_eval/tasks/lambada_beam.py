"""
The LAMBADA dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI
"""
import inspect
import lm_eval.datasets.lambada.lambada
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from datasets import Dataset


_CITATION = """
@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}
"""


class LAMBADA_BEAM(Task):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.lambada.lambada)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        pass

    def validation_docs(self):
        # return self.dataset["validation"]
        return Dataset.from_dict(self.dataset["validation"][:50])

    def test_docs(self):
        pass

    def doc_to_text(self, doc):
        return doc["text"].rsplit(" ", 1)[0]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["text"].rsplit(" ", 1)[1]

    def construct_requests(self, doc, ctx):
        output = rf.greedy_until(ctx, self.doc_to_target(doc))

        return output

    def process_results(self, doc, results):
        output = results[0].split(' ')[0]
        ground_truth = self.doc_to_target(doc).strip()
        
        return {"acc": int(ground_truth in output)}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
