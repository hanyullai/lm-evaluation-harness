import transformers
import torch
from lm_eval.base import BaseLM
import torch.nn as nn
from lm_eval import utils
from tqdm import tqdm


device_map = {'transformer.word_embeddings': 0, 'transformer.layers.0': 0, 'transformer.layers.1': 0, 'transformer.layers.2': 0, 'transformer.layers.3': 0, 'transformer.layers.4': 0, 'transformer.layers.5': 0, 'transformer.layers.6': 0, 'transformer.layers.7': 0, 'transformer.layers.8': 1, 'transformer.layers.9': 1, 'transformer.layers.10': 1, 'transformer.layers.11': 1, 'transformer.layers.12': 1, 'transformer.layers.13': 1, 'transformer.layers.14': 1, 'transformer.layers.15': 1, 'transformer.layers.16': 1, 'transformer.layers.17': 2, 'transformer.layers.18': 2, 'transformer.layers.19': 2, 'transformer.layers.20': 2, 'transformer.layers.21': 2, 'transformer.layers.22': 2, 'transformer.layers.23': 2, 'transformer.layers.24': 2, 'transformer.layers.25': 2, 'transformer.layers.26': 3, 'transformer.layers.27': 3, 'transformer.layers.28': 3, 'transformer.layers.29': 3, 'transformer.layers.30': 3, 'transformer.layers.31': 3, 'transformer.layers.32': 3, 'transformer.layers.33': 3, 'transformer.layers.34': 3, 'transformer.layers.35': 4, 'transformer.layers.36': 4, 'transformer.layers.37': 4, 'transformer.layers.38': 4, 'transformer.layers.39': 4, 'transformer.layers.40': 4, 'transformer.layers.41': 4, 'transformer.layers.42': 4, 'transformer.layers.43': 4, 'transformer.layers.44': 5, 'transformer.layers.45': 5, 'transformer.layers.46': 5, 'transformer.layers.47': 5, 'transformer.layers.48': 5, 'transformer.layers.49': 5, 'transformer.layers.50': 5, 'transformer.layers.51': 5, 'transformer.layers.52': 5, 'transformer.layers.53': 6, 'transformer.layers.54': 6, 'transformer.layers.55': 6, 'transformer.layers.56': 6, 'transformer.layers.57': 6, 'transformer.layers.58': 6, 'transformer.layers.59': 6, 'transformer.layers.60': 6, 'transformer.layers.61': 6, 'transformer.layers.62': 7, 'transformer.layers.63': 7, 'transformer.layers.64': 7, 'transformer.layers.65': 7, 'transformer.layers.66': 7, 'transformer.layers.67': 7, 'transformer.layers.68': 7, 'transformer.layers.69': 7, 'transformer.final_layernorm': 7, 'lm_head': 7}

class GLM130BLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="THUDM/glm-130b",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        
        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained,
            device_map=device_map,
            torch_dtype=torch.half
        )
        
        self.model.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained
        )


        self.vocab_size = self.tokenizer.vocab_size


        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer(string, return_tensors='pt')['input_ids'][0].tolist()

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)[:-2]

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :self.vocab_size]

    def greedy_until(self, requests):
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return 0, x[0]

        re_ord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]
                
            context_enc = self.tokenizer(context + ' [MASK]', return_tensors='pt').to(self.device)['input_ids']

            cont = self._model_generate(
                context_enc
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1]-2:])

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)

    def _model_generate(self, context):

        return self.model.generate(
            input_ids=context, max_new_tokens=5, do_sample=False, num_beams=16
        )