import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def compute_temporal_relations(pl_module, batch, phase='train'):
    relations = {
        'label': batch['label'], 
        'comp': batch['comp']
    }
    labels = torch.LongTensor([int(y == 'after') for y in relations['label']]).cuda()
    infer_multimodal = pl_module.infer(batch)
    predictions = infer_multimodal['predictions']
    loss = F.cross_entropy(predictions, labels)

    preds = [torch.argmax(predictions[i]).item() for i in range(len(predictions))]
    correct = sum([1 for pred, gt in zip(preds, labels) if pred == gt])
    acc = correct / len(predictions)
    pl_module.log(f"{phase}/loss", loss)
    pl_module.log(f"{phase}/accuracy", acc)
    ret = {
        "accuracy": acc,
        "loss": loss,
        "predictions": predictions,
        "labels": batch['label'],
    }
    return ret