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

# 4) Unconditional sampling
with torch.no_grad():
    # T controls DDIM steps; 100 is a good default (matches other configs)
    samples = model.sample(N=16, device=device, T=100)  # returns [0,1]
    grid = torchvision.utils.make_grid(samples, nrow=4)
    torchvision.utils.save_image(grid, os.path.join(OUT_DIR, "samples_T100.png"))
print("Saved:", os.path.join(OUT_DIR, "samples_T100.png"))

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