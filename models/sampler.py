# -*- coding: utf-8 -*-
#
# @File:   sampler.py
# @Author: Haozhe Xie
# @Date:   2023-04-10 13:42:48
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-15 20:14:37
# @Email:  root@haozhexie.com
# @Ref: https://github.com/samb-t/unleashing-transformers

import math
import torch
import torch.nn.functional as F


class AbsorbingDiffusionSampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.denoise_fn = Transformer(cfg)

    def _sample_time(self, bs, device):
        t = torch.randint(
            1, self.cfg.NETWORK.SAMPLER.TOTAL_STEPS + 1, (bs,), device=device
        ).long()
        pt = torch.ones_like(t).float() / self.cfg.NETWORK.SAMPLER.TOTAL_STEPS
        return t, pt

    def _q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (
            t.float().unsqueeze(-1) / self.cfg.NETWORK.SAMPLER.TOTAL_STEPS
        )
        x_t[mask] = self.cfg.NETWORK.VQGAN.N_EMBED
        x_0_ignore[torch.bitwise_not(mask)] = -1

        return x_t, x_0_ignore, mask

    def forward(self, x_0):
        bs, device = x_0.size(0), x_0.device
        # choose what time steps to compute loss at
        t, pt = self._sample_time(bs, device)
        # make x noisy and denoise
        x_t, x_0_ignore, _ = self._q_sample(x_0, t)
        # sample p(x_0 | x_t)
        x_0_hat_logits = self.denoise_fn(x_t, t).permute(0, 2, 1)

        return t, pt, x_0_hat_logits, x_0_ignore

    def sample(
        self, n_samples, sample_steps, x_t=None, temperature=1.0, device="cuda:0"
    ):
        sample_steps = list(range(1, sample_steps + 1))
        if x_t is None:
            x_t = (
                torch.ones(
                    (n_samples, self.cfg.NETWORK.VQGAN.ATTN_RESOLUTION**2),
                    device=device,
                ).long()
                * self.cfg.NETWORK.VQGAN.N_EMBED
            )
        # Initialize: unmasked = torch.zeros_like(x_t, device=device).bool()
        unmasked = x_t != self.cfg.NETWORK.VQGAN.N_EMBED
        for t in reversed(sample_steps):
            t = torch.full((n_samples,), t, device=device, dtype=torch.long)
            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits = self.denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temperature
            x_0_dist = torch.distributions.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        return x_t


class Transformer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = torch.nn.Embedding(
            cfg.NETWORK.VQGAN.N_EMBED + 1, cfg.NETWORK.SAMPLER.N_EMBED
        )
        self.pos_emb = torch.nn.Parameter(
            torch.zeros(1, cfg.NETWORK.SAMPLER.BLOCK_SIZE, cfg.NETWORK.SAMPLER.N_EMBED)
        )
        self.drop = torch.nn.Dropout(cfg.NETWORK.SAMPLER.DROPOUT)

        # Transformer
        self.blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.NETWORK.SAMPLER.N_LAYERS)]
        )
        # decoder head
        self.ln_f = torch.nn.LayerNorm(cfg.NETWORK.SAMPLER.N_EMBED)
        self.head = torch.nn.Linear(
            cfg.NETWORK.SAMPLER.N_EMBED, cfg.NETWORK.VQGAN.N_EMBED, bias=False
        )

        # TODO: It seems that the function is not used in unleashing transformer.
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)
        t = token_embeddings.shape[1]
        assert (
            t <= self.cfg.NETWORK.SAMPLER.BLOCK_SIZE
        ), "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(cfg.NETWORK.SAMPLER.N_EMBED)
        self.ln2 = torch.nn.LayerNorm(cfg.NETWORK.SAMPLER.N_EMBED)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                cfg.NETWORK.SAMPLER.N_EMBED, 4 * cfg.NETWORK.SAMPLER.N_EMBED
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                4 * cfg.NETWORK.SAMPLER.N_EMBED, cfg.NETWORK.SAMPLER.N_EMBED
            ),
            torch.nn.Dropout(cfg.NETWORK.SAMPLER.DROPOUT),
        )

    def forward(self, x, layer_past=None, return_present=False):
        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present

        return x


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.NETWORK.SAMPLER.N_EMBED % cfg.NETWORK.SAMPLER.N_HEAD == 0
        self.n_head = cfg.NETWORK.SAMPLER.N_HEAD
        # key, query, value projections for all heads
        self.key = torch.nn.Linear(
            cfg.NETWORK.SAMPLER.N_EMBED, cfg.NETWORK.SAMPLER.N_EMBED
        )
        self.query = torch.nn.Linear(
            cfg.NETWORK.SAMPLER.N_EMBED, cfg.NETWORK.SAMPLER.N_EMBED
        )
        self.value = torch.nn.Linear(
            cfg.NETWORK.SAMPLER.N_EMBED, cfg.NETWORK.SAMPLER.N_EMBED
        )
        # regularization
        self.attn_drop = torch.nn.Dropout(cfg.NETWORK.SAMPLER.DROPOUT)
        self.resid_drop = torch.nn.Dropout(cfg.NETWORK.SAMPLER.DROPOUT)
        # output projection
        self.proj = torch.nn.Linear(
            cfg.NETWORK.SAMPLER.N_EMBED, cfg.NETWORK.SAMPLER.N_EMBED
        )

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        present = torch.stack((k, v))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present
