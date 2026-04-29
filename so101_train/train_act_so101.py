#!/usr/bin/env python3
"""
SO-101 双臂 ACT 训练（纯 PyTorch，不依赖 lerobot policy 模块）

架构：
  ResNet18(pretrained) → 图像特征
  CVAE encoder         → 隐变量 z（训练时建模多模态，推理时 z=0）
  Transformer encoder  → 融合 [z, state, image]
  Transformer decoder  → 预测 chunk_size 步动作序列

用法：
  cd so101_train
  python train_act_so101.py
  python train_act_so101.py --chunk-size 50 --epochs 2000 --batch 8
  python train_act_so101.py --no-image --chunk-size 30   # 无图快速验证
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).parent.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedHDF5Dataset(Dataset):
    """
    从 robomimic HDF5 加载，输出 ACT 所需的 action chunk 格式。
    """

    def __init__(self, hdf5_path, chunk_size=50, use_image=True):
        self.chunk_size = chunk_size
        self.use_image  = use_image
        self._eps   = []
        self._index = []

        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f["data"].keys()):
                if not key.startswith("demo_"):
                    continue
                obs = f[f"data/{key}/obs"]
                ep = {
                    "state":   np.concatenate([
                        obs["robot0_eef_pos"][:],
                        obs["robot0_eef_quat"][:],
                        obs["robot1_eef_pos"][:],
                        obs["robot1_eef_quat"][:],
                    ], axis=1).astype(np.float32),
                    "actions": f[f"data/{key}/actions"][:].astype(np.float32),
                }
                if use_image and "agentview_image" in obs:
                    raw = obs["agentview_image"][:]
                    ep["images"] = (raw.transpose(0,3,1,2).astype(np.float32) / 255.0)  # (N,3,H,W) fp32，避免__getitem__转换
                ep["n"] = len(ep["state"])
                self._eps.append(ep)
                for t in range(ep["n"]):
                    self._index.append((len(self._eps) - 1, t))

        print(f"Dataset: {len(self._eps)} demos, {len(self._index)} frames, "
              f"chunk={chunk_size}, image={'yes' if use_image else 'no'}")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_i, t = self._index[idx]
        ep      = self._eps[ep_i]
        n       = ep["n"]

        state = torch.from_numpy(ep["state"][t])   # (14,)

        t_end = min(t + self.chunk_size, n)
        raw   = ep["actions"][t:t_end]
        if len(raw) < self.chunk_size:
            pad = np.repeat(ep["actions"][n-1:n], self.chunk_size - len(raw), axis=0)
            raw = np.concatenate([raw, pad], axis=0)
        actions = torch.from_numpy(raw)             # (chunk_size, 10)

        image = None
        if self.use_image and "images" in ep:
            image = torch.from_numpy(ep["images"][t].copy())   # (3,H,W) fp32，已在__init__转好

        return state, image, actions


def collate_fn(batch):
    states, images, actions = zip(*batch)
    out = {
        "state":   torch.stack(states),
        "actions": torch.stack(actions),
    }
    if images[0] is not None:
        out["image"] = torch.stack(images)
    return out


def compute_stats(dataset, n=3000):
    idx = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    acts, states = [], []
    for i in idx:
        s, _, a = dataset[int(i)]
        acts.append(a.numpy())
        states.append(s.numpy())
    acts   = np.stack(acts)    # (n, chunk, 10)
    states = np.stack(states)  # (n, 14)
    return {
        "a_mean": acts.mean(axis=(0, 1)).astype(np.float32),
        "a_std":  acts.std(axis=(0, 1)).clip(1e-6).astype(np.float32),
        "s_mean": states.mean(axis=0).astype(np.float32),
        "s_std":  states.std(axis=0).clip(1e-6).astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ACT Policy（纯 PyTorch 实现）
# ─────────────────────────────────────────────────────────────────────────────

class ResNetEncoder(nn.Module):
    """ResNet18(pretrained) → d_model 维特征向量。"""
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        import torchvision.models as M
        weights = M.ResNet18_Weights.DEFAULT if pretrained else None
        r = M.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(r.children())[:-1])  # (B,512,1,1)
        self.proj     = nn.Linear(512, d_model)

    def forward(self, x):                           # (B,3,H,W)
        return self.proj(self.backbone(x).flatten(1))


class CVAEEncoder(nn.Module):
    """
    训练时编码 (state, action_chunk) → (z_mean, z_logvar)。
    推理时 z=0，跳过此模块。
    """
    def __init__(self, state_dim, action_dim, chunk_size, latent_dim, d_model):
        super().__init__()
        self.state_proj  = nn.Linear(state_dim,  d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embed   = nn.Embedding(chunk_size + 1, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4,
                                         dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)
        self.mean_head   = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, state, actions):
        B, T, _ = actions.shape
        tokens = torch.cat([
            self.state_proj(state).unsqueeze(1),   # (B,1,d)
            self.action_proj(actions),             # (B,T,d)
        ], dim=1)                                  # (B,T+1,d)
        pos    = torch.arange(T + 1, device=state.device)
        tokens = tokens + self.pos_embed(pos)
        cls    = self.transformer(tokens)[:, 0]    # 取第一个 token
        return self.mean_head(cls), self.logvar_head(cls)


class ACTPolicy(nn.Module):
    """
    Action Chunking with Transformers（简化版，适合 20 demos 小数据）。
    """

    def __init__(self, state_dim=14, action_dim=10, chunk_size=50,
                 latent_dim=32, d_model=512, n_heads=8,
                 n_enc=4, n_dec=7, use_image=True, kl_weight=10.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight  = kl_weight
        self.use_image  = use_image

        # 观测编码器
        self.cvae_enc    = CVAEEncoder(state_dim, action_dim, chunk_size, latent_dim, d_model)
        self.state_proj  = nn.Linear(state_dim,  d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        if use_image:
            self.image_enc = ResNetEncoder(d_model, pretrained=True)

        # 主 Transformer
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                               dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)

        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4,
                                               dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)

        # 可学习的动作查询 & 位置编码
        self.action_queries = nn.Embedding(chunk_size, d_model)
        self.enc_pos        = nn.Embedding(16,          d_model)   # 足够容纳所有 encoder token
        self.dec_pos        = nn.Embedding(chunk_size,  d_model)

        # 输出头
        self.output_proj = nn.Linear(d_model, action_dim)

    def _encode(self, state, image, z):
        tokens = [
            self.latent_proj(z),       # (B,d)
            self.state_proj(state),    # (B,d)
        ]
        if self.use_image and image is not None:
            tokens.append(self.image_enc(image))   # (B,d)

        enc_in = torch.stack(tokens, dim=1)        # (B, n_tok, d)
        pos    = torch.arange(enc_in.shape[1], device=state.device)
        enc_in = enc_in + self.enc_pos(pos)
        return self.encoder(enc_in)                # (B, n_tok, d)

    def _decode(self, memory, B):
        pos      = torch.arange(self.chunk_size, device=memory.device)
        queries  = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        queries  = queries + self.dec_pos(pos)
        out      = self.decoder(queries, memory)   # (B, T, d)
        return self.output_proj(out)               # (B, T, action_dim)

    def forward(self, state, image, actions):
        """训练前向：返回 (total_loss, l1_loss, kl_loss)。"""
        z_mean, z_logvar = self.cvae_enc(state, actions)
        z    = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        pred = self._decode(self._encode(state, image, z), state.shape[0])

        l1  = F.l1_loss(pred, actions)
        kl  = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()
        return l1 + self.kl_weight * kl, l1, kl

    @torch.no_grad()
    def predict(self, state, image=None):
        """推理：z=0，返回 (chunk_size, action_dim) 动作序列。"""
        self.eval()
        z    = torch.zeros(state.shape[0], self.latent_dim, device=state.device)
        pred = self._decode(self._encode(state, image, z), state.shape[0])
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 训练入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5",       default="dataset.hdf5")
    ap.add_argument("--chunk-size", type=int,   default=50)
    ap.add_argument("--epochs",     type=int,   default=2000)
    ap.add_argument("--batch",      type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--kl-weight",  type=float, default=10.0)
    ap.add_argument("--d-model",    type=int,   default=256)
    ap.add_argument("--n-dec",      type=int,   default=4)
    ap.add_argument("--no-image",   action="store_true")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--save-every", type=int,   default=200)
    args = ap.parse_args()

    device    = ("cuda" if torch.cuda.is_available() else "cpu") \
                if args.device == "auto" else args.device
    use_image = not args.no_image
    print(f"Device={device}  chunk={args.chunk_size}  d_model={args.d_model}  "
          f"epochs={args.epochs}  image={use_image}")

    # ── 数据集 ──────────────────────────────────────────────────
    dataset = ChunkedHDF5Dataset(HERE / args.hdf5,
                                 chunk_size=args.chunk_size,
                                 use_image=use_image)
    n_workers = min(8, args.batch // 8)  # Linux fork 安全，按 batch 大小自适应
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         collate_fn=collate_fn, num_workers=n_workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(n_workers > 0),
                         prefetch_factor=2 if n_workers > 0 else None)

    # ── 归一化统计 ───────────────────────────────────────────────
    print("Computing normalization stats...")
    stats  = compute_stats(dataset)
    a_mean = torch.tensor(stats["a_mean"], device=device)
    a_std  = torch.tensor(stats["a_std"],  device=device)
    s_mean = torch.tensor(stats["s_mean"], device=device)
    s_std  = torch.tensor(stats["s_std"],  device=device)
    print(f"  action std (mean over dims): {stats['a_std'].mean():.5f}")

    # ── 模型 ────────────────────────────────────────────────────
    policy = ACTPolicy(
        state_dim=14, action_dim=10,
        chunk_size=args.chunk_size,
        latent_dim=32,
        d_model=args.d_model,
        n_heads=4 if args.d_model <= 256 else 8,
        n_enc=4, n_dec=args.n_dec,
        use_image=use_image,
        kl_weight=args.kl_weight,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"ACT parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── 输出目录 ─────────────────────────────────────────────────
    out_dir = HERE / "trained_models" / "act_so101"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 训练循环 ─────────────────────────────────────────────────
    print(f"\nStart training...\n")
    for epoch in range(1, args.epochs + 1):
        policy.train()
        tot = l1_sum = kl_sum = 0.0
        t0  = time.time()

        for batch in loader:
            state   = (batch["state"].to(device)   - s_mean) / s_std
            actions = (batch["actions"].to(device)  - a_mean) / a_std
            image   = batch["image"].to(device) if "image" in batch else None

            optimizer.zero_grad()
            loss, l1, kl = policy(state, image, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            tot    += loss.item()
            l1_sum += l1.item()
            kl_sum += kl.item()

        n = len(loader)
        print(f"Epoch {epoch:5d}  loss={tot/n:.5f}  "
              f"l1={l1_sum/n:.5f}  kl={kl_sum/n:.5f}  "
              f"t={time.time()-t0:.1f}s")

        if epoch % args.save_every == 0:
            ckpt = out_dir / f"act_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch, "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": stats, "args": vars(args),
            }, ckpt)
            print(f"  → Saved {ckpt.name}")

    print("Done.")


if __name__ == "__main__":
    main()
