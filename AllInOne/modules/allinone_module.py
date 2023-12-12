import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
from AllInOne.modules import heads, allinone_utils
from AllInOne.modules import objectives as objectives
from AllInOne.modules.temporal_roll import TemporalRoll
from AllInOne.modules import base_vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

try:
    import clip
except:
    pass


class TemporalRelationsModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.use_clip = config["use_clip"]
        self.save_hyperparameters()
        self.separator_token = 30
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        if self.use_clip:
            self.clip_device = "cuda"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.clip_device, jit=False)
            self.clip_model = self.clip_model.float()
            self.clip_classifier = nn.Sequential(
                nn.Linear(1024, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )

        else:
            self.text_embeddings = BertEmbeddings(bert_config)
            self.text_embeddings.apply(objectives.init_weights)

            self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
            self.token_type_embeddings.apply(objectives.init_weights)
            self.event_order_embeddings = nn.Embedding(2, config["hidden_size"])
            self.event_order_embeddings.apply(objectives.init_weights)

            flag = 0
            if self.hparams.config["load_path"] == "":
                self.transformer = getattr(vit, self.hparams.config["vit"])(
                    pretrained=True, config=self.hparams.config
                )
            else:
                self.transformer = getattr(vit, self.hparams.config["vit"])(
                    pretrained=False, config=self.hparams.config
                )
                # freeze allinone
                # for name, para in self.transformer.named_parameters():
                #     para.requires_grad = False
            self.pooler1 = heads.Pooler(config["hidden_size"])
            self.pooler1.apply(objectives.init_weights)
            self.pooler2 = heads.Pooler(config["hidden_size"])
            self.pooler2.apply(objectives.init_weights)
        self.num_frames = config["num_frames"]

        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.text_embeddings.position_embeddings.weight.size() != state_dict['text_embeddings.position_embeddings.weight'].size():
                state_dict.pop('text_embeddings.position_embeddings.weight', None)
                state_dict.pop('text_embeddings.position_ids', None)
            state_dict = self._inflate_positional_embeds(state_dict)
            self.load_state_dict(state_dict, strict=False)
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            print("====load checkpoint=====")
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = self._inflate_positional_embeds(state_dict)
            self.load_state_dict(state_dict, strict=False)
        self.temporal_roll_module = TemporalRoll(n_segment=self.num_frames, v=0)
        self.temporal_embeddings = nn.Embedding(2, 384)
        self.temporal_module = nn.Sequential(
            nn.Linear(1920, 860),
            nn.GELU(),
            nn.Linear(860, 2)
        )

    def batch_splitter(self, batch): 
        batch_split = [{}, {}]
        for key, value in batch.items():
            for modality in ['text', 'image']:
                for idx in range(2):
                    if key.startswith(f'{modality}_{idx+1}'):
                        n_key = key.replace(f'{modality}_{idx+1}', modality)
                        batch_split[idx][n_key] = value
                        continue
        return batch_split[0], batch_split[1]
    
    def infer(self, batch, mode='train'):
        batch_1, batch_2 = self.batch_splitter(batch)
        x_1, co_masks_1 = self.encoder(batch_1)
        x_2, co_masks_2 = self.encoder(batch_2)
        text_feats, image_feats = 40, 197

        if self.use_clip:
            feats = torch.cat([x_1, x_2])
            pred = self.clip_classifier(feats)
            ret = {
                "cls_feats": feats,
                "predictions": pred,
            }
            return ret
        
        for i, blk in enumerate(self.transformer.blocks):
            text_feats_1, image_feats_1 = x_1[:, :text_feats], x_1[:, text_feats:]
            text_feats_2, image_feats_2 = x_2[:, :text_feats], x_2[:, text_feats: ]
            image_feats_1 = self.temporal_roll_module(image_feats_1, i)
            image_feats_2 = self.temporal_roll_module(image_feats_2, i)
            x_1 = torch.cat((text_feats_1, image_feats_1), dim=1)
            x_2 = torch.cat((text_feats_2, image_feats_2), dim=1)
            x_1, _ = blk(x_1, mask=co_masks_1)
            x_2, _ = blk(x_2, mask=co_masks_2)

        x_1 = self.transformer.norm(x_1)
        x_2 = self.transformer.norm(x_2)
        x_1 = x_1.view(-1, self.num_frames, x_1.size(-2), x_1.size(-1))
        x_2 = x_2.view(-1, self.num_frames, x_2.size(-2), x_2.size(-1))
        x_1 = torch.mean(x_1, dim=1)
        x_2 = torch.mean(x_2, dim=1)
        cls_feats_1 = self.pooler1(x_1)
        cls_feats_2 = self.pooler2(x_2)
        comp = torch.LongTensor([int(r == 'ends') for r in batch["comp"]]).cuda()
        embs = self.temporal_embeddings(comp).squeeze()
        cls_feats_1 = cls_feats_1.squeeze()
        cls_feats_2 = cls_feats_2.squeeze()

        feats = torch.cat([cls_feats_1, cls_feats_2, embs], -1)
        pred = self.temporal_module(feats)
        if pred.dim() < 2:
            pred = pred.unsqueeze(0)
        ret = {
            "cls_feats": feats,
            "predictions": pred,
        }
        return ret
    
    def encoder(self, batch):
        text_ids = batch[f"text_ids"].cuda()
        text_masks = batch[f"text_masks"].cuda()
        text = batch[f"text"]
        img = batch["image"][0]
        if self.use_clip:
            img = torch.mean(img, dim=1)
            text = clip.tokenize(text, truncate=True).to(self.clip_device)
            image_embeds = self.clip_model.encode_image(img)
            text_embeds = self.clip_model.encode_text(text)
            for idx, _ in enumerate(batch["text_mask"]):
                if batch["text_mask"][idx] == 1:
                    text_embeds[idx] = 0
                if batch["image_embeds"][idx] == 1:
                    image_embeds[idx] = 0
            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            return co_embeds, None

        text_embeds = self.text_embeddings(text_ids)
        text_embeds += self.token_type_embeddings(torch.zeros_like(text_masks))
        text_embeds = torch.repeat_interleave(text_embeds, self.num_frames, dim=0)
        text_masks = torch.repeat_interleave(text_masks, self.num_frames, dim=0)
        img = img.contiguous().view(-1, img.size()[2], img.size()[3], img.size()[4])  # btchw to [bt]chw
        image_embeds, image_masks, _, _ = self.transformer.visual_embed(img, max_image_len=-1)
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, 1))
        image_masks = image_masks.cuda()

        frames = self.hparams.config["num_frames"]
        for idx, _ in enumerate(batch["text_mask"]):
            start = frames * idx
            end = frames * (idx + 1)
            if batch["text_mask"][idx] == 1:
                text_embeds[start:end] = 0
                text_masks[start:end] = 0
            if batch["image_mask"][idx] == 1:
                image_embeds[start:end] = 0
                image_masks[start:end] = 0
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)
        return co_embeds, co_masks

    def forward(self, batch, phase='train'):
        return objectives.compute_temporal_relations(self, batch, phase)

    def training_step(self, batch, batch_idx):
        output = self(batch, phase='train')
        return output

    def validation_step(self, batch, batch_idx):
        output = self(batch, phase='val')
        return output

    def test_step(self, batch, batch_idx):
        output = self(batch, phase='test')
        self.log('loss', output['loss'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('accuracy', output['accuracy'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return output

    def configure_optimizers(self):
        return allinone_utils.set_schedule(self)

    def _inflate_positional_embeds(self, new_state_dict, load_temporal_fix='zeros'):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'transformer.temporal_embed' in new_state_dict and 'transformer.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['transformer.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.hparams.config['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    if load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif load_temporal_fix in ['interp', 'bilinear']:
                        mode = 'nearest'
                        if load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(
                            load_temporal_embed,
                            (curr_num_frames, embed_dim), 
                            mode=mode
                        ).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['transformer.temporal_embed'] = new_temporal_embed
        return new_state_dict