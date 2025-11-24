# save as: quick_sample_imagenet.py (run from diffae repo root)
import os
import torch
import torchvision
from templates import imagenet256_autoenc
from experiment import LitModel

# 1) Point to your trained checkpoint
CKPT = "checkpoints/<your_run_name>/last.ckpt"  # e.g., checkpoints/imagenet256_experiment_lowgpu_20251108_094540_autoenc/last.ckpt
OUT_DIR = "quick_samples"
os.makedirs(OUT_DIR, exist_ok=True)

# 2) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Build config/model and load weights
conf = imagenet256_autoenc()  # architecture + samplers
model = LitModel(conf)
state = torch.load(CKPT, map_location="cpu")
print("Loaded step:", state.get("global_step"))
model.load_state_dict(state["state_dict"], strict=False)
model.to(device).eval()

# 4) Unconditional sampling for autoencoder:
#    sample random style -> map to conditioning -> DDIM sampling
from model.unet_autoenc import BeatGANsAutoencModel
with torch.no_grad():
    N = 16
    x_T = torch.randn(N, 3, conf.img_size, conf.img_size, device=device)
    model_ae: BeatGANsAutoencModel = model.ema_model
    # sample style and map to conditioning
    z = torch.randn(N, conf.style_ch, device=device)
    cond = model_ae.noise_to_cond(z)
    # use T=100 for sharper samples
    sampler_T100 = conf._make_diffusion_conf(T=100).make_sampler()
    gen = sampler_T100.sample(model=model_ae, noise=x_T, cond=cond)  # [-1,1]
    gen = (gen + 1) / 2  # [0,1]
    grid = torchvision.utils.make_grid(gen, nrow=4)
    torchvision.utils.save_image(grid, os.path.join(OUT_DIR, "samples_T100_autoenc.png"))
print("Saved:", os.path.join(OUT_DIR, "samples_T100_autoenc.png"))

# 5) Optional: quick reconstruction sanity-check on random noise x_start
#    (uses x_start to guide sampling; shows the pipeline is wired correctly)
with torch.no_grad():
    x_start = torch.randn(8, 3, conf.img_size, conf.img_size, device=device)  # fake inputs in [-1,1]
    # map to [-1,1] explicitly (already gaussian ~ N(0,1), fine for a quick plumbing check)
    # Do a short render
    pred = model.eval_sampler.sample(model=model.ema_model, noise=torch.randn_like(x_start), cond=None, x_start=x_start)
    pred = (pred + 1) / 2
    x_vis = (x_start + 1) / 2
    grid_in = torchvision.utils.make_grid(x_vis, nrow=4)
    grid_out = torchvision.utils.make_grid(pred, nrow=4)
    torchvision.utils.save_image(grid_in, os.path.join(OUT_DIR, "recon_input_fake.png"))
    torchvision.utils.save_image(grid_out, os.path.join(OUT_DIR, "recon_pred_fake.png"))
print("Saved:", os.path.join(OUT_DIR, "recon_input_fake.png"), "and recon_pred_fake.png")