def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    label = [item['label'] for item in batch]

    input_values_padded = pad_sequence(input_values, batch_first=True)

    attention_mask = torch.ones_like(input_values_padded)
    for i, length in enumerate([len(seq) for seq in input_values_padded]):
        attention_mask[i, length:] = 0

    labels = torch.stack(labels)

    return {
        'input_values': input_values_padded,
        'attention_mask': attention_mask,
        'labels': labels,
    }