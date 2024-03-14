import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer
import re

class CustomDataset(Dataset):
    def __init__(self, max_length):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.questions =  []
        self.answers = []

        with open('./dataset/primary_dataset.txt', 'r') as txtfile:
            self.data = txtfile.readlines()
        
        for row in self.data:
            question, answer = row.split(' - ')

            # Convert to lowercase
            question = question.lower()
            answer = answer.lower()

            # Remove special characters
            question = re.sub(r'[^\w\s]', '', question)
            answer = re.sub(r'[^\w\s]', '', answer)

            self.questions.append(question)
            self.answers.append(answer)
        
        print(len(self.questions))
        print(len(self.answers))  

        assert len(self.questions) == len(self.answers)
    
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        max_sequence_length = self.max_length

        # Tokenize and encode input question
        tokenized_input = self.tokenizer.tokenize(question)
        input_indices = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_input))

        # Pad or truncate input sequence
        padded_input_indices = F.pad(input_indices, (0, max_sequence_length - input_indices.size(0)), value=self.tokenizer.pad_token_id)
        input_mask = (padded_input_indices != self.tokenizer.pad_token_id).unsqueeze(0).unsqueeze(1)

        # Tokenize and encode target answer
        tokenized_target = self.tokenizer.tokenize(answer)
        target_indices = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_target))

        # Pad or truncate target sequence
        padded_target_indices = F.pad(target_indices, (0, max_sequence_length - target_indices.size(0)), value=self.tokenizer.pad_token_id)
        target_mask = (padded_target_indices != self.tokenizer.pad_token_id).unsqueeze(0).unsqueeze(1)
        
        return padded_input_indices, padded_target_indices, input_mask, target_mask
