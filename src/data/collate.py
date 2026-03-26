import torch

# Padding all samples to the same length, building attention mask to differentiate real data from padding
def collate_fn(batch):
    print("===== collate_fn started =====")
    print("Batch-Length:", len(batch))

    input_values = [item['input_values'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    raw_audio = [item['raw_audio_for_whisper'] for item in batch]
    labels = [item['label'] for item in batch]
    max_len = max(len(in_value) for in_value in input_values)
    batch_size = len(input_values)
    input_values_padded = torch.zeros(batch_size, max_len, dtype=input_values[0].dtype)

    for i, in_value in enumerate(input_values):
        length = len(in_value)
        input_values_padded[i, :length] = in_value

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long) # Tensor of same size and type as the padded input
    for i, length in enumerate([len(in_value) for in_value in input_values]):
        attention_mask[i, :length] = 1 # Setting the first length positions (real part) to 1 for each sample in batch

    # Stack of all labels in the batch - target (correct class for each sample) for the loss calculation - difference between the predictions and true labels
    labels_tensor = torch.stack(labels)
    print(f"Tensor labels: {labels_tensor}")

    return {
        'input_values': input_values_padded,
        'transcript': transcripts,
        'attention_mask': attention_mask,
        'raw_audio_for_whisper': raw_audio,
        'labels': labels_tensor
    }