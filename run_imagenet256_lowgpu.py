from experiment import train
from templates import imagenet256_autoenc
from templates_latent import imagenet256_autoenc_latent
import argparse
import torch

if __name__ == "__main__":
    # =============================================================================
    # CONFIGURATION FOR IMAGENET TRAINING
    # =============================================================================

    # ===== CLI ARGUMENTS =====
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Base experiment name (suffixes _autoenc/_latent will be added).",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Number of GPUs to use: 1, 2, or 4.",
    )
    args, _ = parser.parse_known_args()

    # ===== STEP 1: CONFIGURE YOUR DATASET =====
    # Set your dataset size based on how many classes you're training on:
    DATASET_SIZE = 2600  # 2-class subset (default)
    # DATASET_SIZE = 13000     # 10-class subset
    # DATASET_SIZE = 130000    # 100-class subset
    # DATASET_SIZE = 1281167   # Full ImageNet-1K

    # ===== STEP 2: CONFIGURE TRAINING LENGTH =====
    # Set number of epochs:
    AUTOENC_EPOCHS = 500
    LATENT_EPOCHS = 500

    # ===== STEP 3: CONFIGURE GPU SETUP =====
    NUM_GPUS = args.num_gpus

    # =============================================================================
    # GRADIENT ACCUMULATION SETUP (maintains effective batch size)
    # =============================================================================
    # Original setup (4 GPUs):
    # - Step 1: batch_size=32, 4 GPUs -> effective batch size = 128
    # - Step 3: batch_size=256, 1 GPU -> effective batch size = 256
    #
    # OPTION 1: Use 2 GPUs
    # - Step 1: batch_size=32, 2 GPUs, accum_batches=2 -> effective batch size = 128
    # - Step 3: batch_size=128, 1 GPU, accum_batches=2 -> effective batch size = 256
    #
    # OPTION 2: Use 1 GPU (slower but works)
    # - Step 1: batch_size=32, 1 GPU, accum_batches=4 -> effective batch size = 128
    # - Step 3: batch_size=64, 1 GPU, accum_batches=4 -> effective batch size = 256
    # =============================================================================

    print(f"\n{'=' * 70}")
    print("Training Configuration Summary:")
    print(f"{'=' * 70}")
    print(f"  Dataset size:        {DATASET_SIZE:,} images")
    print(f"  Autoencoder epochs:  {AUTOENC_EPOCHS}")
    print(f"  Latent DPM epochs:   {LATENT_EPOCHS}")
    print(f"  Number of GPUs:      {NUM_GPUS}")
    print(f"{'=' * 70}\n")

    assert torch.cuda.device_count() >= NUM_GPUS, (
        f"Requested {NUM_GPUS} GPUs, but only {torch.cuda.device_count()} available."
    )

    if NUM_GPUS == 4:
        # Step 1: train the autoencoder with 4 GPUs
        print("\n" + "=" * 70)
        print("STEP 1: Training Autoencoder (4 GPUs)")
        print("=" * 70)
        gpus = [0, 1, 2, 3]
        conf = imagenet256_autoenc(
            dataset_size=DATASET_SIZE, target_epochs=AUTOENC_EPOCHS
        )
        # Keep global batch small to avoid OOM; match 1-GPU memory (local ~8)
        # Global batch = 32, 4 GPUs => local batch = 8; accum 4 => effective 128
        conf.accum_batches = 4
        # Ensure global batch is 32 (default), in case templates change
        conf.batch_size = 32
        if args.name:
            conf.name = f"{args.name}_autoenc"
        local_bs = conf.batch_size // len(gpus)
        print(f"Global batch: {conf.batch_size} (local {local_bs} × {len(gpus)} GPUs), "
              f"accum: {conf.accum_batches} ⇒ effective: {conf.batch_size_effective}")
        train(conf, gpus=gpus)

        # Step 2: infer latents for training the latent DPM
        print("\n" + "=" * 70)
        print("STEP 2: Inferring Latents")
        print("=" * 70)
        gpus = [0, 1, 2, 3]
        conf.eval_programs = ["infer"]
        train(conf, gpus=gpus, mode="eval")

        # Step 3: train the latent DPM with 1 GPU
        print("\n" + "=" * 70)
        print("STEP 3: Training Latent DPM (1 GPU)")
        print("=" * 70)
        gpus = [0]
        conf = imagenet256_autoenc_latent(
            dataset_size=DATASET_SIZE, target_epochs=LATENT_EPOCHS
        )
        conf.batch_size = 128  # Reduce batch size
        conf.accum_batches = 2  # Accumulate gradients over 2 batches (effective 256)
        if args.name:
            conf.name = f"{args.name}_latent"
        print(
            f"Effective batch size: {conf.batch_size} × 1 GPU × {conf.accum_batches} accum = {conf.batch_size * conf.accum_batches}"
        )
        train(conf, gpus=gpus)

        # Step 4: unconditional sampling + FID evaluation
        print("\n" + "=" * 70)
        print("STEP 4: Evaluation")
        print("=" * 70)
        gpus = [0, 1, 2, 3]
        conf.eval_programs = ["fid(10,10)"]
        train(conf, gpus=gpus, mode="eval")

    elif NUM_GPUS == 2:
        # Step 1: train the autoencoder with 2 GPUs
        print("\n" + "=" * 70)
        print("STEP 1: Training Autoencoder (2 GPUs)")
        print("=" * 70)
        gpus = [0, 1]
        conf = imagenet256_autoenc(
            dataset_size=DATASET_SIZE, target_epochs=AUTOENC_EPOCHS
        )
        conf.scale_up_gpus(2)  # Scale for 2 GPUs instead of 4
        conf.accum_batches = 2  # Accumulate gradients over 2 batches
        if args.name:
            conf.name = f"{args.name}_autoenc"
        print(
            f"Effective batch size: {conf.batch_size} × 2 GPUs × 2 accum = {conf.batch_size_effective}"
        )
        train(conf, gpus=gpus)

        # Step 2: infer latents for training the latent DPM
        print("\n" + "=" * 70)
        print("STEP 2: Inferring Latents")
        print("=" * 70)
        gpus = [0, 1]
        conf.eval_programs = ["infer"]
        train(conf, gpus=gpus, mode="eval")

        # Step 3: train the latent DPM with 1 GPU
        print("\n" + "=" * 70)
        print("STEP 3: Training Latent DPM (1 GPU)")
        print("=" * 70)
        gpus = [0]
        conf = imagenet256_autoenc_latent(
            dataset_size=DATASET_SIZE, target_epochs=LATENT_EPOCHS
        )
        conf.batch_size = 128  # Reduce batch size
        conf.accum_batches = 2  # Accumulate gradients over 2 batches
        if args.name:
            conf.name = f"{args.name}_latent"
        print(
            f"Effective batch size: {conf.batch_size} × 1 GPU × 2 accum = {conf.batch_size * 2}"
        )
        train(conf, gpus=gpus)

        # Step 4: unconditional sampling + FID evaluation
        print("\n" + "=" * 70)
        print("STEP 4: Evaluation")
        print("=" * 70)
        gpus = [0, 1]
        conf.eval_programs = ["fid(10,10)"]
        train(conf, gpus=gpus, mode="eval")

    elif NUM_GPUS == 1:
        # Step 1: train the autoencoder with 1 GPU
        print("\n" + "=" * 70)
        print("STEP 1: Training Autoencoder (1 GPU)")
        print("=" * 70)
        gpus = [0]
        conf = imagenet256_autoenc(
            dataset_size=DATASET_SIZE, target_epochs=AUTOENC_EPOCHS
        )
        conf.batch_size = 8
        conf.batch_size_eval = 8
        conf.accum_batches = 16  # Accumulate gradients over 4 batches
        conf.make_model_conf()
        if args.name:
            conf.name = f"{args.name}_autoenc"

        print(
            f"Effective batch size: {conf.batch_size} × 1 GPU × {conf.accum_batches} accum = {conf.batch_size_effective}"
        )
        train(conf, gpus=gpus)

        # Step 2: infer latents for training the latent DPM
        print("\n" + "=" * 70)
        print("STEP 2: Inferring Latents")
        print("=" * 70)
        gpus = [0]
        conf.eval_programs = ["infer"]
        train(conf, gpus=gpus, mode="eval")

        # Step 3: train the latent DPM with 1 GPU
        print("\n" + "=" * 70)
        print("STEP 3: Training Latent DPM (1 GPU)")
        print("=" * 70)
        gpus = [0]
        conf = imagenet256_autoenc_latent(
            dataset_size=DATASET_SIZE, target_epochs=LATENT_EPOCHS
        )
        # # conf.batch_size = 64  # Reduce batch size
        # # conf.accum_batches = 4  # Accumulate gradients over 4 batches
        conf.batch_size = 16  # Reduce batch size
        conf.batch_size_eval = 16  # Reduce batch size
        conf.accum_batches = 16  # Accumulate gradients over 4 batches
        if args.name:
            conf.name = f"{args.name}_latent"
        # print(f"Effective batch size: {conf.batch_size} × 1 GPU × {conf.accum_batches} accum = {conf.batch_size * conf.accum_batches}")
        # train(conf, gpus=gpus)

        # Step 4: unconditional sampling + FID evaluation
        print("\n" + "=" * 70)
        print("STEP 4: Evaluation")
        print("=" * 70)
        gpus = [0]
        conf.eval_num_images = min(1000, DATASET_SIZE)
        conf.eval_programs = ["fid(10,10)"]
        train(conf, gpus=gpus, mode="eval")
    else:
        raise ValueError(f"NUM_GPUS must be 1, 2 or 4, got {NUM_GPUS}")
