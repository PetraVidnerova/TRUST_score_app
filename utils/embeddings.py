import logging 
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from accelerate import dispatch_model


logger = logging.getLogger("__main__")

@torch.no_grad()
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (B, T, H)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / sum_mask

class Embeddings():

    def __init__(self, device="auto", batch_size=8):

        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base', device_map=device)
        self.model.load_adapter("allenai/specter2",
                                source="hf",
                                load_as="specter2",
                                set_active=True
                                )
                                #device_map="auto") # device map does not work here
        #self.model.to("cuda")

        # this shows some params are still on cpu 
        # for name, param in self.model.named_parameters():
        #     if param.device.type == "cpu":
        #         print("CPU param:", name)
        # exit()

        # so we force them to move on appropriate gpus
        device_map = self.model.hf_device_map
        self.model = dispatch_model(self.model, device_map=device_map)
        self.model.set_active_adapters("specter2")

        print("Active adapters:", self.model.active_adapters)

    def embed_batch(self, text_batch): 
        inputs = self.tokenizer(text_batch, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False, max_length=512)
       
        device = next(self.model.parameters()).device # which gpu to move the inputs to   
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]

            #embedding = mean_pooling(outputs, inputs["attention_mask"])
            #embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        return embedding
    
    def embed(self, titles_abstracts, titles_only=False):
        
        if titles_only:
            logger.debug("Embedding titles only")
            text_batch = [
                title
                for title, abstract in titles_abstracts
            ]
        else:
            text_batch = [
                title + self.tokenizer.sep_token + abstract
                for title, abstract in titles_abstracts
            ]

        if self.batch_size is not None:
            batchsize = self.batch_size
        else:
            batchsize = 1
        batches = [
            text_batch[i:i+batchsize]
            for i in range(0, len(text_batch), batchsize)
        ]
        embeddings = []
        i = 0
        for b in batches:
            embeddings.append(
                self.embed_batch(b)
            )
            i += 1
            yield (
                f"Embeddings calculated for {len(embeddings)*batchsize}/{len(text_batch)} items..."
                + "👻" * (len(batches)-i)
            )
        result = torch.vstack(embeddings)
        yield result
