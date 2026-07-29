# I'm Something of a Painter Myself — CycleGAN

A **CycleGAN** implementation in PyTorch for the Kaggle competition
[I'm Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started):
translating ordinary photographs into Monet-style paintings using **unpaired** image data.

**Placed 2nd of 93 teams.**

The interesting constraint is that no photo↔painting pairs exist — the model never sees a
"correct answer" for any given photo. CycleGAN solves this with two generators and a
**cycle-consistency** requirement: a photo translated to Monet and back again must reconstruct
the original.

---

## Architecture

Two generators (`photo → Monet`, `Monet → photo`) and two PatchGAN discriminators, trained
adversarially.

**Generators** — both variants are implemented and selectable at runtime:

- **U-Net** (default) — encoder/decoder with skip connections between layer *i* and *n−i*,
  which preserve spatial detail and yield sharper output.
- **ResNet** — the original CycleGAN paper's architecture,
  `c7s1-64 → d128 → d256 → R256 × 9 → u128 → u64 → c7s1-3`, using 9 residual blocks at 256×256.

**Discriminators** — **70×70 PatchGAN**. Rather than classifying a whole image as real or fake,
it classifies overlapping patches, which pushes the generator toward convincing high-frequency
texture instead of globally plausible mush.

**Losses** (`losses/`):

| Loss | Weight | Purpose |
|---|---:|---|
| Adversarial (LSGAN) | 1.0 | Realism. Least-squares variant — more stable than vanilla BCE |
| Cycle consistency | 10.0 | `G_BA(G_AB(x)) ≈ x` — the core unpaired-training constraint |
| Identity | 5.0 | `G_AB(y) ≈ y` for images already in the target domain; preserves colour |

`vanilla` and `hinge` adversarial losses are also implemented and switchable via config.

**Training stabilisation**

- **Replay buffer** (size 50) — discriminators train against a history of previously generated
  images, not just the latest batch, which damps oscillation.
- **Linear LR decay** — constant `2e-4` for 15 epochs, then decayed linearly to zero over the
  remaining 15.
- **Adam** with `β = (0.5, 0.999)`, batch size 1, as in the original paper.

---

## Repository structure

```
configs/config.py         All hyperparameters in one dataclass, plus environment presets
models/
  generator_unet.py       U-Net generator (skip connections)
  generator_resnet.py     ResNet generator (9 residual blocks)
  discriminator.py        70x70 PatchGAN discriminator
losses/losses.py          Adversarial / cycle / identity losses
data/dataset.py           Unpaired dataset, transforms, dataloaders
utils/utils.py            Replay buffer, checkpointing, LR scheduler, visualisation
train.py                  Training loop with WandB logging
inference.py              Generates images and packages submission zip
```

Configuration is centralised in a single `Config` dataclass, with presets for different
environments — `get_unet_config()`, `get_resnet_config()`, `get_colab_config()` and
`get_debug_config()` (2 epochs, no WandB) for quick iteration.

---

## Usage

```bash
pip install torch torchvision pillow wandb
```

Point `data_root` in `configs/config.py` at the competition data
(expects `monet_jpg/` and `photo_jpg/` subdirectories).

**Train:**

```bash
python train.py --generator unet          # or: --generator resnet
python train.py --debug --no-wandb        # quick smoke test, 2 epochs
python train.py --resume checkpoints/checkpoint_epoch_20.pth
```

**Generate submission** (the competition expects 7,000–10,000 images):

```bash
python inference.py \
    --generator checkpoints/generator_AB_final.pth \
    --photos /path/to/photo_jpg \
    --type unet \
    --num-images 7500 \
    --zip images.zip
```

Training runs are logged to **Weights & Biases** (project `cyclegan-monet`); pass `--no-wandb`
to disable.
