import torch

def collate_fn(batch):
    print("===== collate_fn started =====")
    print("Batch-Length:", len(batch))

    input_values = [item['input_values'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    raw_audio = [item['raw_audio_for_whisper'] for item in batch]
    labels = [item['label'] for item in batch]
    max_len = max(len(iv) for iv in input_values)


    batch_size = len(input_values)
    input_values_padded = torch.zeros(batch_size, max_len, dtype=input_values[0].dtype)

    for i, iv in enumerate(input_values):
        length = len(iv)
        input_values_padded[i, :length] = iv

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, length in enumerate([len(iv) for iv in input_values]):
        attention_mask[i, :length] = 1

    labels_tensor = torch.stack(labels)
    print(f"Tensor labels: {labels_tensor}")



    return {
        'input_values': input_values_padded,
        'transcript': transcripts,
        'attention_mask': attention_mask,
        'raw_audio_for_whisper': raw_audio,
        'labels': labels_tensor
    }