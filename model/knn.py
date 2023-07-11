import sys
import torch
import datasets

from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SearchEngine(object):

    def __init__(self, dataset_path, model_checkpoint):

        print(f'load model from {model_checkpoint}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.model.to(device)

        print(f'load dataset from {dataset_path}')
        dataset = datasets.load_from_disk(dataset_path=dataset_path)

        print(f'calculates embeddings')
        self.embeddings_dataset = dataset.map(
            lambda x: {"embeddings": SearchEngine.get_embeddings(x["text"]).detach().cpu().numpy()[0]}
        )
        print(f'apply faiss index')
        self.embeddings_dataset.add_faiss_index(column="embeddings")

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    @staticmethod
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, text: str):
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return SearchEngine.cls_pooling(model_output)

    def search(self, query: str, k=5):
        print(f'calculate embeddings for query text')
        query_embedding = SearchEngine.get_embeddings([query]).cpu().detach().numpy()
        print(f'get {k} nearest samples from dataset')
        scores, samples = self.embeddings_dataset.get_nearest_examples("embeddings", query_embedding, k=k)
        print([f'{sample} {scores[i]}' for i, sample in enumerate(samples)])
        return scores, samples

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset path> <model checkpoint>')
        exit(1)
    dataset_path = sys.argv[1]
    model_checkpoint = sys.argv[2]
    search_engine = SearchEngine(dataset_path=dataset_path, model_checkpoint=model_checkpoint)

    while True:
        query = input('type search query:')
        res = search_engine.search(query, k=5)
        print(f'{query} : {res}')

    search_engine.search()
