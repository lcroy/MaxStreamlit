import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGeneration:

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def text_gen(self, text=''):

        len_org = len(text)
        tokens = self.tokenizer.encode(text)

        for i in range(20):
            outputs, _ = self.model(torch.tensor([tokens]))
            next_token = torch.argmax(outputs[0, -1])
            tokens.append(next_token)

        result = self.tokenizer.decode(tokens)
        # result_remove_enter_qua = result[len_org+1:].replace("\n", "").replace('"',"")
        # result_clean = re.sub(r'(\?|\!|\,)', '.', result_remove_enter_qua).split('.')[:-1][0]

        return result


