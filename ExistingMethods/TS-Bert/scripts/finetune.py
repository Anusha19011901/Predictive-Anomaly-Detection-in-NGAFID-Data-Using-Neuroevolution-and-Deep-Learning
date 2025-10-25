# add this at the very top of each script in scripts/
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, yaml, torch, pytorch_lightning as pl, torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryAveragePrecision
from tsbert.model import TimeSeriesBERT
from tsbert.data import make_loader

class FinetuneModule(pl.LightningModule):
    def __init__(self, cfg, ckpt_path=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.m = TimeSeriesBERT(
            d_in=cfg["num_features"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], ff_mult=cfg["ff_mult"], dropout=cfg["dropout"],
            max_len=cfg["max_len"]
        )
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            self.load_state_dict(state, strict=False)  # load encoder & projection
        self.lr = cfg["lr"]; self.wd = cfg["weight_decay"]
        # Data
        self.train_loader = make_loader(cfg["train_npz"], "train", cfg["batch_size"], True, False)
        self.val_loader   = make_loader(cfg["train_npz"], "val",   cfg["batch_size"], False, False)
        self.test_loader  = make_loader(cfg["train_npz"], "test",  cfg["batch_size"], False, False)
        # Metrics
        self.auroc = BinaryAUROC()
        self.f1    = BinaryF1Score()
        self.ap    = BinaryAveragePrecision()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt

    def _step(self, batch, stage):
        x, y = batch
        logits = self.m.classify(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_auroc", self.auroc(probs, y.int()))
        self.log(f"{stage}_f1",    self.f1(probs, y.int()))
        self.log(f"{stage}_ap",    self.ap(probs, y.int()))
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    def train_dataloader(self): return self.train_loader
    def val_dataloader(self):   return self.val_loader
    def test_dataloader(self):  return self.test_loader

if __name__ == "__main__":
    import argparse; parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Existing Methods/TS-Bert/configs/base.yaml")
    parser.add_argument("--ckpt", default="Existing Methods/TS-Bert/checkpoints/tsbert_pretrained.ckpt")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pl.seed_everything(42)
    mod = FinetuneModule(cfg, ckpt_path=args.ckpt if os.path.exists(args.ckpt) else None)
    trainer = pl.Trainer(max_epochs=cfg["max_epochs_finetune"], precision=cfg["precision"],
                         default_root_dir=cfg["logdir"], devices=cfg["devices"])
    trainer.fit(mod)
    trainer.test(mod)
    # Save fine-tuned weights
    os.makedirs(cfg["ckptdir"], exist_ok=True)
    trainer.save_checkpoint(os.path.join(cfg["ckptdir"], "tsbert_finetuned.ckpt"))
