from templates import (
    TrainConfig,
    pretrain_bedroom128,
    pretrain_ffhq128_autoenc130M,
    pretrain_ffhq256_autoenc,
    pretrain_horse128,
    pretrain_celeba64d2c_72M,
    pretrain_imagenet256_autoenc,
)
from choices import (
    TrainMode,
    GenerativeType,
    LossType,
    ModelMeanType,
    ModelVarType,
    OptimizerType,
    Activation,
)
from model.latentnet import LatentNetType
from config import PretrainConfig


def latent_diffusion_config(conf: TrainConfig):
    conf.batch_size = 128
    conf.train_mode = TrainMode.latent_diffusion
    conf.latent_gen_type = GenerativeType.ddim
    conf.latent_loss_type = LossType.mse
    conf.latent_model_mean_type = ModelMeanType.eps
    conf.latent_model_var_type = ModelVarType.fixed_large
    conf.latent_rescale_timesteps = False
    conf.latent_clip_sample = False
    conf.latent_T_eval = 20
    conf.latent_znormalize = True
    conf.total_samples = 96_000_000
    conf.sample_every_samples = 400_000
    conf.eval_every_samples = 20_000_000
    conf.eval_ema_every_samples = 20_000_000
    conf.save_every_samples = 2_000_000
    return conf


def latent_diffusion128_config(conf: TrainConfig):
    conf = latent_diffusion_config(conf)
    conf.batch_size_eval = 32
    return conf


def latent_mlp_2048_norm_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.skip
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_activation = Activation.silu
    conf.net_latent_num_hid_channels = 2048
    conf.net_latent_use_norm = True
    conf.net_latent_condition_bias = 1
    return conf


def latent_mlp_2048_norm_20layers(conf: TrainConfig):
    conf = latent_mlp_2048_norm_10layers(conf)
    conf.net_latent_layers = 20
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_256_batch_size(conf: TrainConfig):
    conf.batch_size = 256
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 2_000_000
    conf.total_samples = 301_000_000
    return conf


def latent_512_batch_size(conf: TrainConfig):
    conf.batch_size = 512
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 5_000_000
    conf.total_samples = 501_000_000
    return conf


def latent_2048_batch_size(conf: TrainConfig):
    conf.batch_size = 2048
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.save_every_samples = 20_000_000
    conf.total_samples = 1_501_000_000
    return conf


def adamw_weight_decay(conf: TrainConfig):
    conf.optimizer = OptimizerType.adamw
    conf.weight_decay = 0.01
    return conf


def ffhq128_autoenc_latent():
    conf = pretrain_ffhq128_autoenc130M()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = "const0.008"
    conf.name = "ffhq128_autoenc_latent"
    return conf


def ffhq256_autoenc_latent():
    conf = pretrain_ffhq256_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = "const0.008"
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = "ffhq256_autoenc_latent"
    return conf


def horse128_autoenc_latent():
    conf = pretrain_horse128()
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 2_001_000_000
    conf.latent_beta_scheduler = "const0.008"
    conf.latent_loss_type = LossType.l1
    conf.name = "horse128_autoenc_latent"
    return conf


def bedroom128_autoenc_latent():
    conf = pretrain_bedroom128()
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 2_001_000_000
    conf.latent_beta_scheduler = "const0.008"
    conf.latent_loss_type = LossType.l1
    conf.name = "bedroom128_autoenc_latent"
    return conf


def celeba64d2c_autoenc_latent():
    conf = pretrain_celeba64d2c_72M()
    conf = latent_diffusion_config(conf)
    conf = latent_512_batch_size(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = adamw_weight_decay(conf)
    # just for the name
    conf.continue_from = PretrainConfig("200M", f"log-latent/{conf.name}/last.ckpt")
    conf.postfix = "_300M"
    conf.total_samples = 301_000_000
    conf.latent_beta_scheduler = "const0.008"
    conf.latent_loss_type = LossType.l1
    conf.name = "celeba64d2c_autoenc_latent"
    return conf


def imagenet256_autoenc_latent(dataset_size=2600, target_epochs=100):
    """
    ImageNet 256x256 latent diffusion configuration.

    Args:
        dataset_size: Number of images in your dataset (same as autoencoder)
        target_epochs: Number of epochs to train on latents (default: 100)

    The total_samples will be automatically calculated based on the number
    of latent codes (equal to dataset_size) and desired epochs.

    Examples:
        # For 2-class training:
        conf = imagenet256_autoenc_latent(dataset_size=2600, target_epochs=100)

        # For full ImageNet:
        conf = imagenet256_autoenc_latent(dataset_size=1281167, target_epochs=80)
    """
    conf = pretrain_imagenet256_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)

    # Calculate total_samples for latent training
    # Note: latent dataset size = number of encoded images = dataset_size
    conf.total_samples = dataset_size * target_epochs

    # Set evaluation frequency based on dataset size
    if dataset_size < 10_000:
        # Small subset (2-10 classes)
        conf.eval_every_samples = min(50_000, conf.total_samples // 4)
        conf.eval_ema_every_samples = min(50_000, conf.total_samples // 4)
        conf.sample_every_samples = min(25_000, conf.total_samples // 8)
        conf.save_every_samples = min(50_000, conf.total_samples // 4)
    elif dataset_size < 100_000:
        # Medium subset (10-100 classes)
        conf.eval_every_samples = 1_000_000
        conf.eval_ema_every_samples = 1_000_000
        conf.sample_every_samples = 200_000
        conf.save_every_samples = 2_000_000
    else:
        # Large dataset (100+ classes or full ImageNet)
        conf.eval_every_samples = 20_000_000
        conf.eval_ema_every_samples = 20_000_000
        conf.sample_every_samples = 4_000_000
        conf.save_every_samples = 10_000_000

    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = "const0.008"
    conf.name = "imagenet256_autoenc_latent"

    # Print configuration summary
    print(f"\n{'=' * 70}")
    print("ImageNet256 Latent Diffusion Configuration:")
    print(f"{'=' * 70}")
    print(f"  Latent dataset size: {dataset_size:,} (one per image)")
    print(f"  Target epochs:       {target_epochs}")
    print(f"  Total samples:       {conf.total_samples:,}")
    print(f"  Batch size:          {conf.batch_size}")
    print(f"  Eval every:          {conf.eval_every_samples:,} samples")
    print(f"  Save every:          {conf.save_every_samples:,} samples")
    print(f"{'=' * 70}\n")

    return conf
