import torch

def collate_fn(batch):
    print("===== collate_fn started =====")
    print("Batch-Type:", type(batch))
    print("Batch-Length:", len(batch))

    input_values = [item['input_values'] for item in batch]
    labels = [item['label'] for item in batch]

    print(f"Anzahl Items: {len(input_values)}")
    print(f"Längen: {[len(iv) for iv in input_values]}")
    print(f"Labels: {labels}")

    max_len = max(len(iv) for iv in input_values)
    print(f"max_len: {max_len}")

    batch_size = len(input_values)
    input_values_padded = torch.zeros(batch_size, max_len, dtype=input_values[0].dtype)
    print(f"Zero-Tensor erstellt: {input_values_padded.shape}")

    for i, iv in enumerate(input_values):
        length = len(iv)
        print(f"Verarbeite Item {i}, Länge {length}")
        input_values_padded[i, :length] = iv
        print(f"Item {i} eingefügt")

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, length in enumerate([len(iv) for iv in input_values]):
        attention_mask[i, :length] = 1
    print(f"Attention-Mask erstellt: {attention_mask.shape}")

    labels_tensor = torch.stack(labels)
    print(f"Labels-Tensor: {labels_tensor}")

    print("===== collate_fn finished =====")

    return {
        'input_values': input_values_padded,
        'attention_mask': attention_mask,
        'labels': labels_tensor
    }