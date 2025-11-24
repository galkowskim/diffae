# save as: quick_sample_imagenet.py (run from diffae repo root)
import os
import torch
import torchvision
from templates import imagenet256_autoenc
from experiment import LitModel
from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

# 1) Point to your trained checkpoint
CKPT = "/mnt/evafs/groups/ganzha_23/mgalkowski/copy_testing/gcd/gcd/last.ckpt"
OUT_DIR = "quick_samples"
os.makedirs(OUT_DIR, exist_ok=True)

# 2) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Build config/model and load weights
conf = imagenet256_autoenc()  # architecture + samplers
model = LitModel(conf)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
print("Loaded step:", state.get("global_step"))
model.load_state_dict(state["state_dict"], strict=False)
model.to(device).eval()

from model.unet_autoenc import BeatGANsAutoencModel
with torch.no_grad():
    # Reconstruction of real images from HF ImageNet with higher T to reduce blur
    N = 16
    model_ae: BeatGANsAutoencModel = model.ema_model
    T_RENDER = 200
    sampler_render = conf._make_diffusion_conf(T=T_RENDER).make_sampler()
    # Load a small labeled subset (e.g., classes 339/340) to keep results meaningful
    tfm = T.Compose([T.Resize(conf.img_size), T.CenterCrop(conf.img_size), T.ToTensor()])
    imgs_01 = []
    ds = load_dataset("imagenet-1k", split="train", streaming=True, cache_dir=os.getenv("DATASET_CACHE", None), token=os.environ["HF_TOKEN"])
    for ex in ds:
        # if you want arbitrary images, remove this filter
        if not (ex.get('label', None) in (339, 340)):
            continue
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        imgs_01.append(tfm(img))
        if len(imgs_01) == N:
            break
    x01 = torch.stack(imgs_01).to(device)         # [0,1]
    x_m1p1 = (x01 - 0.5) * 2                      # [-1,1]
    # Deterministic forward (zeros noise) tends to be sharper/stabler
    x_T = torch.zeros_like(x_m1p1)
    gen = sampler_render.sample(model=model_ae, noise=x_T, cond=None, x_start=x_m1p1)
    gen = (gen + 1) / 2
    grid_in = torchvision.utils.make_grid(x01, nrow=4)
    grid_out = torchvision.utils.make_grid(gen, nrow=4)
    out_in = os.path.join(OUT_DIR, "recon_input.png")
    out_out = os.path.join(OUT_DIR, "recon_output_T200.png")
    torchvision.utils.save_image(grid_in, out_in)
    torchvision.utils.save_image(grid_out, out_out)
    print("Saved:", out_in, "and", out_out)

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