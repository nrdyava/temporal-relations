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

def compute_temporal_relations(
    pl_module, 
    batch, 
    git=False, 
    git_processor=None, 
    phase='train'
):
    if git:
        assert git_processor is not None
    relations = {
        'label': batch['label'], 
        'comp': batch['comp']
    }
    labels = torch.LongTensor([int(y == 'after') for y in relations['label']]).cuda()
    infer_multimodal = pl_module.infer(batch)
    if git:
        loss_fn = nn.CrossEntropyLoss()
        text, output, labels, need_predict = infer_multimodal
        preds, need_logits, need_labels = [], [], []
        # loss = output.loss
        logits = output.logits[:, -512:]
        ids_ = torch.argmax(torch.softmax(logits, dim=2), dim=2).cpu()
        for logit, ids, label, idxs in zip(logits, ids_, labels, need_predict):
            need_logits.append(logit[idxs[0]-1:idxs[1]])
            need_labels.append(label[idxs[0]:idxs[1]+1])
            pred = git_processor.decode(ids[idxs[0]-1])
            preds.append(pred)
        need_logits = torch.vstack(need_logits)
        need_labels = torch.vstack(need_labels)
        need_labels = torch.reshape(need_labels, (-1,))
        loss = loss_fn(need_logits, need_labels)
        correct = sum([1 for pred, gt in zip(preds, batch['label']) if pred == gt])
        acc = correct / len(preds)
        ret = {
            "accuracy": acc,
            "loss": loss,
        }
    else:
        predictions = infer_multimodal['predictions']
        # print('labels', labels.shape)
        loss = F.cross_entropy(predictions, labels)

        preds = [torch.argmax(predictions[i]).item() for i in range(len(predictions))]
        correct = sum([1 for pred, gt in zip(preds, labels) if pred == gt])
        acc = correct / len(predictions)
        ret = {
            "accuracy": acc,
            "loss": loss,
            # "predictions": predictions,
            # "labels": batch['label'],
        }
    pl_module.log(f"{phase}/loss", loss)
    pl_module.log(f"{phase}/accuracy", acc)
    return ret