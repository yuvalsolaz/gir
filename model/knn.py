import sys

import faiss
import torch
import datasets

from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SearchEngine(object):

    def __init__(self, dataset_path, similarity_model, index_column="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT):
        print(f'load model from {similarity_model}')
        self.tokenizer = AutoTokenizer.from_pretrained(similarity_model)
        self.model = AutoModel.from_pretrained(similarity_model)
        self.model.to(device)

        print(f'load dataset from: {dataset_path}')
        dataset = datasets.load_from_disk(dataset_path=dataset_path)['train']# .select(range(10000))

        print(f'calculates embeddings')
        self.embeddings_dataset = dataset.map(
            lambda x: {"embeddings": self.get_embeddings(x["english_desc"]).detach().cpu().numpy()[0]},
            batched=False
        )
        print(f'add faiss index: {index_column} metric type: {metric_type}')
        self.embeddings_dataset.add_faiss_index(index_name=index_column, column=index_column, metric_type=metric_type)

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
        model_output = self.model(**encoded_input)
        cls_vector = SearchEngine.cls_pooling(model_output)
        return torch.divide(cls_vector,cls_vector.norm())

    def search(self, query: str, k=5):
        print(f'calculate embeddings for: {query}')
        query_embedding = self.get_embeddings(text=[query]).cpu().detach().numpy()
        print(f'get {k} dataset nearest samples for: {query}')
        scores, samples = self.embeddings_dataset.get_nearest_examples("embeddings", query_embedding, k=k)
        results = ['{} {:.2f}'.format(sample, scores[i]) for i, sample in enumerate(samples["english_desc"])]
        print('\n'.join(results))
        return scores, samples

    def inference(self, sentence: str):
        scores, samples = self.search(query=sentence, k=5)
        return samples['cell_id'][0], scores[0]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage: python {sys.argv[0]} <dataset path> <model checkpoint>')
        exit(1)
    dataset_path = sys.argv[1]
    model_checkpoint = sys.argv[2]
    search_engine = SearchEngine(dataset_path=dataset_path, similarity_model=model_checkpoint,
                                 index_column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)

    while True:
        query = input('type search query (or exit for quit):')
        if query.lower() == 'exit':
            print('bye bye...')
            break
        scores, samples = search_engine.search(query, k=5)
