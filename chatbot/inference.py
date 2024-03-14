import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import Transformer
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_closest_text(sample_text, text_list):
    # Vectorize the texts
    vectorizer = TfidfVectorizer()
    text_matrix = vectorizer.fit_transform([sample_text] + text_list)
    
    # Calculate cosine similarity between the sample text and each text in the list
    similarities = cosine_similarity(text_matrix)[0][1:]
    
    # Find the index of the most similar text
    closest_index = np.argmax(similarities)
    
    return closest_index

def greedy_decode(model, src_input, src_mask):
    model.eval()
    with torch.no_grad():
        output = model(src_input, src_mask)
    return output

def tensor_to_text(output_tensor, tokenizer):
    # Apply argmax to get the indices of the most probable token
    _, predicted_indices = output_tensor.max(dim=-1)
    # Convert indices back to tokens
    output_texts = []
    probs = []
    for sequence_indices in predicted_indices:
        tokens = tokenizer.convert_ids_to_tokens(sequence_indices.tolist())
        # Remove special tokens and join tokens to form the output text
        output_text = ' '.join(token for token in tokens if token not in ['<start>', '<end>', tokenizer.pad_token])
        output_texts.append(output_text)
        # Calculate probabilities
        softmax_probs = F.softmax(output_tensor, dim=-1)
        sequence_probs = softmax_probs.max(dim=-1)[0]
        probs.append(sequence_probs.mean().item())
    return output_texts, probs


def inference_on_model(message):

    input_text = message
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    vocab_size = 30522
    embedding_size = 512
    num_heads = 8
    hidden_size = 2048
    num_layers = 6

    model = Transformer(vocab_size, embedding_size, num_heads, hidden_size, num_layers)

    model.load_state_dict(torch.load('./chatbot/weights/best_checkpoint.pth.tar', map_location='cpu'))
    print('Model weights have been loaded..')

    # Tokenize input text
    tokenized_input = tokenizer.tokenize(input_text)
    input_indices = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)])
    input_mask = (input_indices != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

    # Perform inference
    output_indices = greedy_decode(model, input_indices, input_mask)
    output_text, probs = tensor_to_text(output_indices, tokenizer)

    return output_text, probs


def read_secondary_database():
    questions =  []
    answers = []

    with open('./chatbot/datasets/secondary_dataset.txt', 'r') as txtfile:
        data = txtfile.readlines()

    for row in data:
        question, answer = row.split(' - ')

        # Convert to lowercase
        question = question.lower()
        answer = answer.lower()

        question = re.sub(r'[^\w\s]', '', question)
        answer = re.sub(r'[^\w\s]', '', answer)

        questions.append(question)
        answers.append(answer)
    
    return questions, answers

def inference(message=''):

    response, confidence_score = inference_on_model(message)

    if confidence_score[0] < 0.8:
        questions, answers = read_secondary_database()
        index = find_closest_text(message,questions)
        response = answers[index]

        # Write the data on primary database
        with open('./chatbot/datasets/primary_dataset.txt', 'a') as txtfile:
            txtfile.write(message+' - '+response)
    
    print('Input : ',message)
    print('Response : ', response)

    return response

if __name__ == "__main__":

    message = 'Who has taken the most wickets in Test cricket?'
    inference(message)