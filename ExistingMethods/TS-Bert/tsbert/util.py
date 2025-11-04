import torch, random
def make_span_mask(B, L, ratio=0.15, span=8, device="cpu"):
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    to_mask = int(L * ratio)
    for b in range(B):
        covered = 0
        while covered < to_mask:
            start = random.randint(0, max(0, L-span))
            mask[b, start:start+span] = True
            covered = mask[b].sum().item()
    return mask

def apply_mask(x, proj, mask_token, mask_bool):
    # x: [B, L, D_in] -> project to d_model via proj? No; we feed mask token before encoder.
    # Replace input at masked positions with a learned mask token (after input_proj in module.forward)
    # Here we simply stuff a flag; the module will handle via input_proj+mask_token trick in trainer.
    return x
