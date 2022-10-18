import transformers
import torch
from lm_eval.base import BaseLM
import torch.nn as nn


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
            device_map='auto',
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

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            input_ids=context, max_new_tokens=max_length, eos_token_id=eos_token_id, do_sample=False, num_beams=16
        )