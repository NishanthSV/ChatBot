import math
import torch
import torch.nn as nn
from model import Transformer
from transformers import BertTokenizer
from dataloader import CustomDataset
from torch.utils.data import DataLoader
import time

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
    for sequence_indices in predicted_indices:
        tokens = tokenizer.convert_ids_to_tokens(sequence_indices.tolist())
        # Remove special tokens and join tokens to form the output text
        output_text = ' '.join(token for token in tokens if token not in ['<start>', '<end>', tokenizer.pad_token])
        output_texts.append(output_text)
    return output_texts


with open('training_log.txt','w+') as file:
    file.write(f"Training logs..\n")

# Hyperparameters
vocab_size = 30522
embedding_size = 512
num_heads = 8
hidden_size = 2048
num_layers = 6
max_length = 100
batch_size = 64

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create Transformer model
model = Transformer(vocab_size, embedding_size, num_heads, hidden_size, num_layers)

# Initialize optimizer with initial learning rate
initial_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Step Decay Learning Rate
lr_decay_rate = 0.96
lr_decay_epochs = 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Prepare data loader
dataset = CustomDataset(max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 500
best_loss = float('inf')
early_stopping_patience = 80
early_stopping_counter = 0
for epoch in range(num_epochs):
    
    for param_group in optimizer.param_groups:
        print("Epoch:", epoch+1, "- Learning Rate:", round(param_group['lr'], 6), end=' ')

    start_time = time.time()
    total_loss = 0
    for src, trg, src_mask, trg_mask in dataloader:
        # src, trg, src_mask, trg_mask = src.cuda(), trg.cuda(), src_mask.cuda(), trg_mask.cuda()
        
        optimizer.zero_grad()
        
        output = model(src, src_mask)
        
        output = output.view(-1, vocab_size).type(torch.FloatTensor)
        target = trg.view(-1).type(torch.LongTensor)

        loss = criterion(output, target)
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Loss: {avg_loss}")

    # Learning Rate Scheduling
    scheduler.step()

    print('Time Taken : ', time.time()-start_time)

    # Early Stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stopping_counter = 0
    

        # Save model checkpoint
        torch.save(model.state_dict(), f"best_checkpoint.pth.tar")
        print('Model is updated...')

        # Save training log
        with open('training_log.txt','a') as file:
            file.write(f"Epoch {epoch+1}, Loss: {round(avg_loss, 4)}\n")

        # Sample inference
        input_text = 'Who has taken the most wickets in Test cricket?'
        tokenized_input = tokenizer.tokenize(input_text)
        input_indices = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)])
        input_mask = (input_indices != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        output_indices = greedy_decode(model, input_indices, input_mask)
        output_text = tensor_to_text(output_indices, tokenizer)[0]

        print("Input:", input_text)
        print("Output:", output_text)
        print()

        with open('training_log.txt','a') as file:
            file.write(f"Sample Inference - Epoch {epoch+1}\n")
            file.write("Input: " + input_text + "\n")
            file.write("Output: " + output_text + "\n\n")
    
    else:
        early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
