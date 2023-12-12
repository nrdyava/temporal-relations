import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import DataCollatorForLanguageModeling, BertTokenizer


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.num_frames = _config["num_frames"]
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.masking_type = _config["masking_type"]
        self.masking_prob = _config["masking_prob"]

        collator = DataCollatorForLanguageModeling
        self.tokenizer = get_pretrained_tokenizer('bert-base-uncased')
        self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=True)
        self.vocab_size = self.tokenizer.vocab_size
        _config["vocab_size"] = self.vocab_size
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer
            self.setup_flag = True
    
    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            split="train",
            num_frames=self.num_frames,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            masking_type=self.masking_type,
            masking_prob=self.masking_prob,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            split="val",
            num_frames=self.num_frames,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            masking_type=self.masking_type,
            masking_prob=self.masking_prob,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            split="test",
            num_frames=self.num_frames,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            masking_type=self.masking_type,
            masking_prob=self.masking_prob,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return loader
