import torch

def collate_fn(batch):
    print("===== collate_fn gestartet =====")
    print("Batch-Typ:", type(batch))
    print("Batch-Länge:", len(batch))

    if len(batch) > 0:
        # Greife auf das erste Item zu
        first = batch[0]
        print("Item0 keys:", first.keys())
        print("Item0 input_values shape:", first['input_values'].shape)
        print("Item0 label:", first['label'])

    # Erstelle einen winzigen Tensor, um zu sehen, ob torch funktioniert
    x = torch.tensor([1.0])
    print("Tensor erstellt:", x)

    return {"dummy": x}