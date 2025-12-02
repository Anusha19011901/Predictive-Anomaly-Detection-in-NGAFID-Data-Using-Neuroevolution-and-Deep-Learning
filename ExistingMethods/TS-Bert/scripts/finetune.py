# =====================================================================
# TS-BERT FINETUNING SCRIPT (NAB ANOMALY CLASSIFICATION)
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

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryAveragePrecision,
)

# Internal modules
from tsbert.model import TimeSeriesBERT
from tsbert.data import make_loader


# =====================================================================
# Finetuning Lightning Module
# =====================================================================
class FinetuneModule(pl.LightningModule):
    def __init__(self, cfg, ckpt_path=None):
        super().__init__()
        self.save_hyperparameters(cfg)

        # ---------------------------
        # Base TS-BERT encoder
        # ---------------------------
        self.m = TimeSeriesBERT(
            d_in=cfg["num_features"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            ff_mult=cfg["ff_mult"],
            dropout=cfg["dropout"],
            max_len=cfg["max_len"],
        )

        # If a pretrained checkpoint is provided, load it (encoder weights)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            # Lightning checkpoints usually have a "state_dict"
            if isinstance(state, dict) and "state_dict" in state:
                self.load_state_dict(state["state_dict"], strict=False)
            else:
                # Fallback: assume raw model state dict
                self.m.load_state_dict(state, strict=False)

        # ---------------------------
        # Classification head
        # ---------------------------
        self.classifier = torch.nn.Linear(cfg["d_model"], 1)

        # ---------------------------
        # Safe hyperparameters (cast to float)
        # ---------------------------
        self.lr = float(cfg.get("lr", 3e-4))
        self.wd = float(cfg.get("weight_decay", 1e-4))

        npz_path = cfg["train_npz"]
        bs = cfg["batch_size"]

        # make_loader(npz_path, split, batch_size, shuffle, for_pretrain)
                # OLD:
        # self.train_loader = make_loader(npz_path, "train", bs, shuffle=True)
        # self.val_loader   = make_loader(npz_path, "val",   bs, shuffle=False)

        # Replace with:
        self.train_loader = make_loader(
            npz_path,
            "train",
            bs,
            shuffle=True,
            oversample_anomalies=True,   # <-- key change
        )

        self.val_loader = make_loader(
            npz_path,
            "val",
            bs,
            shuffle=False,
            oversample_anomalies=False,  # explicit, for clarity
        )

        # ---------------------------
        # Metrics
        # ---------------------------
        self.auroc = BinaryAUROC()
        self.f1    = BinaryF1Score()
        self.ap    = BinaryAveragePrecision()

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    # Dataloaders
    def train_dataloader(self): return self.train_loader
    def val_dataloader(self):   return self.val_loader

    # -----------------------------------------------------------------
    # Forward pass: sequence -> encoder -> pooled -> logit
    # -----------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, L, D]
        """
        h = self.m.input_proj(x)
        h = self.m.pos(h)
        h = self.m.encoder(h)         # [B, L, d_model]

        # Simple temporal pooling (mean over time)
        h_pool = h.mean(dim=1)        # [B, d_model]

        logit = self.classifier(h_pool).squeeze(-1)  # [B]
        return logit

    # -----------------------------------------------------------------
    # Training step (supervised binary classification)
    # -----------------------------------------------------------------
    def training_step(self, batch, _):
        x, y = batch

        # BCE wants float labels
        y_float = y.view(-1).float()

        # Metrics want int labels
        y_int = y.view(-1).long()

        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y_float)

        probs = torch.sigmoid(logits)

        # Logging loss
        self.log("train_loss", loss, prog_bar=True)

        # Logging metrics
        self.log("train_auroc", self.auroc(probs, y_int), prog_bar=False, on_epoch=True)
        self.log("train_f1",    self.f1(probs, y_int),    prog_bar=False, on_epoch=True)
        self.log("train_ap",    self.ap(probs, y_int),    prog_bar=False, on_epoch=True)

        return loss

    # -----------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------
    def validation_step(self, batch, _):
        x, y = batch

        y_float = y.view(-1).float()
        y_int   = y.view(-1).long()

        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y_float)
        probs = torch.sigmoid(logits)

        auroc = self.auroc(probs, y_int)
        f1    = self.f1(probs, y_int)
        ap    = self.ap(probs, y_int)

        self.log("val_loss",  loss,  prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_f1",    f1,    prog_bar=True)
        self.log("val_ap",    ap,    prog_bar=True)


# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ExistingMethods/TS-Bert/configs/base.yaml")
    parser.add_argument(
        "--ckpt",
        default="ExistingMethods/TS-Bert/checkpoints/tsbert_pretrained.ckpt",
        help="Path to pretrained TS-BERT checkpoint.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    pl.seed_everything(42)

    module = FinetuneModule(cfg, ckpt_path=args.ckpt)

    ckptdir = cfg["ckptdir"]
    os.makedirs(ckptdir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs_finetune"],
        precision=cfg["precision"],
        devices=cfg["devices"],
        default_root_dir=cfg["logdir"],
    )

    trainer.fit(module)

    trainer.save_checkpoint(
        os.path.join(ckptdir, "tsbert_finetuned.ckpt")
    )
