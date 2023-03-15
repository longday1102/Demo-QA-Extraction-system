import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def read_data(data_dir):
    with open(data_dir, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    return data

class Create_datasetDict:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def preprocessing(self):
        contexts = []
        questions = []
        answers = []
        ids = []
        for data in self.dataset['data']:
            for para in data['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append({'answer_start': [answer['answer_start']], 'text': [answer['text']]})
                        ids.append(qa['id'])
        icqa = [ids, contexts, questions, answers] 
        return icqa  
    
    def __call__(self, test_size):
        icqa = self.preprocessing()
        data = {'id': icqa[0], 'context': icqa[1], 'question': icqa[2], 'answer': icqa[3]}
        
        # Building dataset with format same to SQuAD 2.0 
        df = pd.DataFrame(data = data)
        train_df, valid_df = train_test_split(df, test_size = test_size, random_state = 1, shuffle = True)
        
        train_dict = Dataset.from_dict(train_df)
        valid_dict = Dataset.from_dict(valid_df)
        dataset_dict = DatasetDict({'train': train_dict, 'validation': valid_dict})
        return dataset_dict
