from transformers import AutoTokenizer
import torch
import os


class Aspect_Classifier():
    def __init__(self, model_path=None, token_path=None):
        self.model, self.tokenizer = self._get_model_and_tokenizer(
            model_path, token_path)

    def _test_model(self, model, tokenizer):
        model.eval()
        with torch.no_grad():
            sentence = self.clean_text("the milk was not that great.")
            phrase = self.clean_text("milk")

            tok = tokenizer(
                sentence,
                phrase,
                return_tensors='pt',
                return_token_type_ids="True"
            )
            probablities = torch.softmax(model(**tok).logits, dim=1)
            output = torch.argmax(probablities)
            assert output.item() == 0

    def _get_model_and_tokenizer(self, model_path=None, tokenizer_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.getcwd(),
                os.path.join(
                    *[os.pardir, "weights", "tiny_bert.pt"]
                ))
        if tokenizer_path is None:
            tokenizer_path = os.path.join(
                os.getcwd(),
                os.path.join(
                    *[os.pardir, "weights", "tiny_bert_tokenizer"]
                ))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = torch.load(model_path)
        self._test_model(model, tokenizer)
        return model, tokenizer

    def clean_text(self, sentence):
        punctuation = '!"#$%()*+-/:;<=>@[\\]^_`{|}~'
        sentence = "".join(ch for ch in sentence if ch not in set(punctuation))
        sentence = sentence.replace("[0-9]", " ")
        sentence = " ".join(sentence.split()).lower()
        return sentence

    def __call__(self, text, phrase):
        text = self.clean_text(text)
        phrase = self.clean_text(phrase)
        with torch.no_grad():
            tok = self.tokenizer(
                text,
                phrase,
                return_tensors='pt',
                return_token_type_ids="True"
            )
            probablities = torch.softmax(self.model(**tok).logits, dim=1)
            output = torch.argmax(probablities)
            probablities = probablities.tolist()[0]
            return {
                "label": output.item(),
                "negative": probablities[0],
                "neutral": probablities[1],
                "positive": probablities[2],
            }
