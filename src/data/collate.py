import torch

"""
collate_fn function to pad all samples to the same length, building attention mask to differentiate real data from padding
Parameters: 
    batch: list of samples
Returns: 
    collated batch
"""
def collate_fn(batch):

    input_values = [item['input_values'] for item in batch]
    transcript_ids = torch.stack([item['transcript_ids'] for item in batch])
    transcript_mask = torch.stack([item['transcript_mask'] for item in batch])
    labels = [item['label'] for item in batch]


    max_len = max(len(in_value) for in_value in input_values)
    batch_size = len(input_values)
    input_values_padded = torch.zeros(batch_size, max_len, dtype=input_values[0].dtype)

    for i, in_value in enumerate(input_values):
        length = len(in_value)
        input_values_padded[i, :length] = in_value

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long) # Tensor of the same size and type as the padded input
    for i, length in enumerate([len(in_value) for in_value in input_values]):
        attention_mask[i, :length] = 1 # Setting the first length positions (real part) to 1 for each sample in the batch

    # Stack of all labels in the batch - target (correct class for each sample) for the loss calculation - difference between the predictions and true labels
    labels_tensor = torch.stack(labels)
    attack_idxs = torch.stack([item['attack_idx'] for item in batch])
    
    return {
        'input_values': input_values_padded,
        'transcript_ids': transcript_ids,
        'transcript_mask': transcript_mask,
        'attention_mask': attention_mask,
        'labels': labels_tensor,
        'attack_idx': attack_idxs
    }