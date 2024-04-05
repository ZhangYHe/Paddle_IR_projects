import paddle
from paddle.io import Dataset, DataLoader
import pandas as pd
import numpy as np
from paddlenlp.transformers import BertModel, BertTokenizer

class MS_MARCO_Dataset(Dataset):
    def __init__(self, queries_file, qrels_file, collection_file, tokenizer_name='bert-base-uncased'):
        self.queries = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])
        self.qrels = pd.read_csv(qrels_file, sep='\t', header=None, names=['qid', '0', 'pid', 'rel'])
        self.collection = pd.read_csv(collection_file, sep='\t', header=None, names=['pid', 'passage'])
        self.all_pids = self.collection['pid'].tolist()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        qrel_row = self.qrels.iloc[idx]
        qid = qrel_row['qid']
        positive_pid = qrel_row['pid']
        
        query_text = self.queries[self.queries['qid'] == qid]['query'].values[0]
        positive_passage = self.collection[self.collection['pid'] == positive_pid]['passage'].values[0]

        
        negative_pid = np.random.choice([pid for pid in self.all_pids if pid != positive_pid])
        negative_passage = self.collection[self.collection['pid'] == negative_pid]['passage'].values[0]
        # print("qid", qid)
        # print("Query Text:", query_text)
        # print("Positive Passage:", positive_passage)
        # print("Negative Passage:", negative_passage)
        # print("\n")

        # Tokenize and convert to tensors
        query_encoding = self.tokenizer(query_text, return_tensors='pd', padding='max_length', truncation=True, max_length=128, return_attention_mask=True)
        positive_passage_encoding = self.tokenizer(positive_passage, return_tensors='pd', padding='max_length', truncation=True, max_length=128, return_attention_mask=True)
        negative_passage_encoding = self.tokenizer(negative_passage, return_tensors='pd', padding='max_length', truncation=True, max_length=128, return_attention_mask=True)

        return {
            "input_ids_query": query_encoding['input_ids'].squeeze(0),  # Remove batch dimension
            "attention_mask_query": query_encoding['attention_mask'].squeeze(0),  # Remove batch dimension
            "input_ids_pos": positive_passage_encoding['input_ids'].squeeze(0),  # Remove batch dimension
            "attention_mask_pos": positive_passage_encoding['attention_mask'].squeeze(0),  # Remove batch dimension
            "input_ids_neg": negative_passage_encoding['input_ids'].squeeze(0),  # Remove batch dimension
            "attention_mask_neg": negative_passage_encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        }


if __name__ == "__main__":
    # Example usage
    collection_file = '/home/zhangyh/dataset/MS MARCO/passage raranking/collection.tsv'
    queries_file = '/home/zhangyh/dataset/MS MARCO/passage raranking/queries.train.tsv'
    qrels_file = '/home/zhangyh/dataset/MS MARCO/passage raranking/qrels.train.tsv'

    dataset = MS_MARCO_Dataset(queries_file=queries_file, qrels_file=qrels_file, collection_file=collection_file)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(loader):
        if i==2:
            break
        print(data)

    print(len(dataset))
    sample = dataset[0]
    print(sample)

