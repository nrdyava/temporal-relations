import os
import yaml
import copy
import torch
import shutil
import datetime
import pytorch_lightning as pl
import torch.distributed as dist
from AllInOne.modules import TemporalRelationsModel
from AllInOne.datamodules.multitask_datamodule import MTDataModule

torch.autograd.set_detect_anomaly(False)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), 
        torch.zeros(s, s, s, s, device=dev),
    )

def main(config_path):
    force_cudnn_initialization()
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    dm = MTDataModule(_config, dist=True)
    model = TemporalRelationsModel(_config)

    exp_name = f'{_config["exp_name"]}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/accuracy",
        mode="max",
        save_last=True,
    )
    now = datetime.datetime.now()
    instance_name = f'{exp_name}_{now.year}_{now.month}_{now.day}'
    log_dir = os.path.join(_config["log_dir"], instance_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=instance_name,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    num_gpus = _config["num_gpus"]
    
    shutil.copyfile(config_path, os.path.join(log_dir, "config.yaml"))
    print('='*20 + ' Config: ' + '='*20)
    print(instance_name)
    print(_config)
    print('='*50)

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        gradient_clip_val=0.0001
    )
    print("accumulate grad batches is: ", trainer.accumulate_grad_batches)
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        model.eval()
        # model.freeze()
        trainer.test(model, datamodule=dm)