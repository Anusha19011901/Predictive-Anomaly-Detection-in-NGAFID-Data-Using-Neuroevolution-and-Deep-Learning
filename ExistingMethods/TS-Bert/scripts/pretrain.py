# add this at the very top of each script in scripts/
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, yaml, torch, pytorch_lightning as pl, torch.nn.functional as F
from torch.utils.data import DataLoader
from tsbert.model import TimeSeriesBERT
from tsbert.data import make_loader
from tsbert.utils import make_span_mask

class PretrainModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.m = TimeSeriesBERT(
            d_in=cfg["num_features"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], ff_mult=cfg["ff_mult"], dropout=cfg["dropout"],
            max_len=cfg["max_len"]
        )
        self.mask_ratio = cfg["mask_ratio"]; self.mask_span = cfg["mask_span"]
        self.lr = cfg["lr"]; self.wd = cfg["weight_decay"]
        self.train_loader = make_loader(cfg["train_npz"], "train", cfg["batch_size"], True, True)
        self.val_loader   = make_loader(cfg["train_npz"], "val",   cfg["batch_size"], False, True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}

    def train_dataloader(self): return self.train_loader
    def val_dataloader(self):   return self.val_loader

    def _mask_inputs(self, x):
        B, L, D = x.shape
        mask_bool = make_span_mask(B, L, self.mask_ratio, self.mask_span, x.device)  # [B,L]
        x_in = x.clone()
        # Replace masked positions with learned mask token after projection:
        # we can't inject after projection here, so hack: add a channel of flags
        # Simpler: do it inside model by adding mask token at encoder input:
        # Build encoder input by mixing projected x and mask token
        return mask_bool, x_in

    def training_step(self, batch, _):
        x, _ = batch  # [B,L,D]
        mask_bool, x_in = self._mask_inputs(x)
        # Prepare encoder input: project x, then overwrite masked positions with mask token vector
        h = self.m.input_proj(x_in)
        h = self.m.pos(h)
        B,L,Dm = h.shape
        mask_tok = self.m.mask_token.expand(B, L, -1)
        h = torch.where(mask_bool.unsqueeze(-1), mask_tok, h)
        h = self.m.encoder(h)
        ypred = self.m.to_value(h)  # [B,L,D_in]
        loss = F.mse_loss(ypred[mask_bool], x[mask_bool])
        self.log("train_mse_mask", loss)
        return loss

    def validation_step(self, batch, _):
        x, _ = batch
        mask_bool, x_in = self._mask_inputs(x)
        h = self.m.input_proj(x_in); h = self.m.pos(h)
        B,L,Dm = h.shape
        mask_tok = self.m.mask_token.expand(B, L, -1)
        h = torch.where(mask_bool.unsqueeze(-1), mask_tok, h)
        h = self.m.encoder(h)
        ypred = self.m.to_value(h)
        loss = F.mse_loss(ypred[mask_bool], x[mask_bool])
        self.log("val_mse_mask", loss, prog_bar=True)

if __name__ == "__main__":
    import argparse; parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Existing Methods/TS-Bert/configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pl.seed_everything(42)
    mod = PretrainModule(cfg)
    ckptdir = cfg["ckptdir"]; os.makedirs(ckptdir, exist_ok=True)
    trainer = pl.Trainer(max_epochs=cfg["max_epochs_pretrain"], precision=cfg["precision"],
                         default_root_dir=cfg["logdir"], devices=cfg["devices"])
    trainer.fit(mod)
    trainer.save_checkpoint(os.path.join(ckptdir, "tsbert_pretrained.ckpt"))
