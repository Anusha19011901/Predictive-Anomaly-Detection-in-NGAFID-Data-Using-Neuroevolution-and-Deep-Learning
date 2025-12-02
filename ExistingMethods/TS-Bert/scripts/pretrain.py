# =====================================================================
# TS-BERT PRETRAINING SCRIPT (CLEAN + FIXED + SAFE FOR MPS / MACOS)
# =====================================================================

# Ensure imports from tsbert/ work
import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Core imports
import yaml
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Internal project modules
from tsbert.model import TimeSeriesBERT
from tsbert.data import make_loader
from tsbert.util import make_span_mask


# =====================================================================
# Pretraining Lightning Module
# =====================================================================
class PretrainModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # ---------------------------
        # Model architecture
        # ---------------------------
        self.m = TimeSeriesBERT(
            d_in=cfg["num_features"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            ff_mult=cfg["ff_mult"],
            dropout=cfg["dropout"],
            max_len=cfg["max_len"]
        )

        # ---------------------------
        # Safe-cast hyperparameters (protects against YAML string errors)
        # ---------------------------
        self.lr = float(cfg.get("lr", 3e-4))
        self.wd = float(cfg.get("weight_decay", 1e-4))

        # Masking parameters
        self.mask_ratio = float(cfg["mask_ratio"])
        self.mask_span  = int(cfg["mask_span"])

        # ---------------------------
        # Loaders
        # ---------------------------
        npz_path = cfg["train_npz"]
        bs = cfg["batch_size"]

        self.train_loader = make_loader(npz_path, "train", bs, True, True)
        self.val_loader   = make_loader(npz_path, "val",   bs, False, True)


    # -----------------------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    # Lightning requires these explicitly since we override DataLoader logic
    def train_dataloader(self): return self.train_loader
    def val_dataloader(self):   return self.val_loader

    # -----------------------------------------------------------------
    # Masking logic
    # -----------------------------------------------------------------
    def _mask_inputs(self, x):
        B, L, D = x.shape
        mask_bool = make_span_mask(
            B, L, self.mask_ratio, self.mask_span, device=x.device
        )  # shape [B,L]
        return mask_bool, x

    # -----------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------
    def training_step(self, batch, _):
        x, _ = batch  # x = [B, L, D]

        mask_bool, x_in = self._mask_inputs(x)

        # Input projection + positional encoding
        h = self.m.input_proj(x_in)
        h = self.m.pos(h)

        B, L, Dm = h.shape
        mask_tok = self.m.mask_token.expand(B, L, -1)

        # Replace masked positions with mask token
        h = torch.where(mask_bool.unsqueeze(-1), mask_tok, h)

        # Encode
        h = self.m.encoder(h)

        # Predict original values
        ypred = self.m.to_value(h)

        # MSE only on masked positions
        loss = F.mse_loss(ypred[mask_bool], x[mask_bool])

        self.log("train_mse_mask", loss)
        return loss

    # -----------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------
    def validation_step(self, batch, _):
        x, _ = batch
        mask_bool, x_in = self._mask_inputs(x)

        h = self.m.input_proj(x_in)
        h = self.m.pos(h)

        B, L, Dm = h.shape
        mask_tok = self.m.mask_token.expand(B, L, -1)
        h = torch.where(mask_bool.unsqueeze(-1), mask_tok, h)
        h = self.m.encoder(h)

        ypred = self.m.to_value(h)
        loss = F.mse_loss(ypred[mask_bool], x[mask_bool])

        self.log("val_mse_mask", loss, prog_bar=True)


# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ExistingMethods/TS-Bert/configs/base.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    pl.seed_everything(42)

    module = PretrainModule(cfg)

    # Ensure checkpoint directory exists
    ckptdir = cfg["ckptdir"]
    os.makedirs(ckptdir, exist_ok=True)

    # Trainer (MacOS MPS safe: precision=32-true)
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs_pretrain"],
        precision=cfg["precision"],
        devices=cfg["devices"],
        default_root_dir=cfg["logdir"]
    )

    # Run training
    trainer.fit(module)

    # Save checkpoint
    trainer.save_checkpoint(
        os.path.join(ckptdir, "tsbert_pretrained.ckpt")
    )
