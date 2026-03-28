"""
Test script to verify GPU setup and memory before running full training.
Run this first to make sure your configuration will work.
"""

import torch
from templates import imagenet256_autoenc
from templates_latent import imagenet256_autoenc_latent


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_gpu_availability():
    print_section("GPU Availability Check")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. You need a NVIDIA GPU with CUDA.")
        return False

    num_gpus = torch.cuda.device_count()
    print("✅ CUDA is available")
    print(f"✅ Found {num_gpus} GPU(s)")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Memory: {memory_gb:.2f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")

    return True


def estimate_memory_requirements(batch_size, img_size=256):
    """Rough estimate of memory per batch"""
    print_section("Memory Estimate")

    # Model parameters (rough estimate for DiffAE ImageNet256)
    model_params = 130e6  # ~130M parameters
    model_memory_gb = (model_params * 4) / 1024**3  # 4 bytes per param (fp32)

    # Activations and gradients (very rough estimate)
    # For a 256x256 image with UNet architecture
    activation_memory_per_sample = (
        img_size**2 * 3 * 4 * 50
    ) / 1024**3  # rough estimate
    batch_memory_gb = activation_memory_per_sample * batch_size

    # Total (very rough)
    total_estimate = model_memory_gb * 2 + batch_memory_gb  # x2 for model + gradients

    print("Estimated memory per GPU:")
    print(f"  - Model (params + optimizer): ~{model_memory_gb * 2:.2f} GB")
    print(f"  - Batch ({batch_size} samples): ~{batch_memory_gb:.2f} GB")
    print(f"  - Total estimate: ~{total_estimate:.2f} GB")
    print("\n⚠️  This is a ROUGH estimate. Actual usage may vary ±50%")

    return total_estimate


def test_configuration(num_gpus, accum_batches, batch_size, stage_name):
    """Test a specific configuration"""
    print(f"\n{stage_name}:")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Gradient accumulation: {accum_batches}")
    effective_batch = batch_size * num_gpus * accum_batches
    print(f"  ✅ Effective batch size: {effective_batch}")

    # Memory estimate
    estimate_memory_requirements(batch_size)

    return effective_batch


def test_all_configurations():
    print_section("Configuration Tests")

    # Original 4 GPU config
    print("\n--- ORIGINAL (4 GPUs) ---")
    eff1 = test_configuration(4, 1, 32, "Step 1: Autoencoder")
    eff2 = test_configuration(1, 1, 256, "Step 3: Latent DPM")

    # 2 GPU config
    print("\n\n--- MODIFIED (2 GPUs) ---")
    eff1_2gpu = test_configuration(2, 2, 32, "Step 1: Autoencoder")
    eff2_2gpu = test_configuration(1, 2, 128, "Step 3: Latent DPM")

    # 1 GPU config
    print("\n\n--- MODIFIED (1 GPU) ---")
    eff1_1gpu = test_configuration(1, 4, 32, "Step 1: Autoencoder")
    eff2_1gpu = test_configuration(1, 4, 64, "Step 3: Latent DPM")

    # Verify effective batch sizes match
    print_section("Verification")
    checks = [
        (eff1, eff1_2gpu, eff1_1gpu, "Autoencoder", 128),
        (eff2, eff2_2gpu, eff2_1gpu, "Latent DPM", 256),
    ]

    all_good = True
    for orig, two_gpu, one_gpu, name, expected in checks:
        if orig == two_gpu == one_gpu == expected:
            print(
                f"✅ {name}: All configurations match (effective batch size = {expected})"
            )
        else:
            print(f"❌ {name}: Mismatch detected!")
            print(
                f"   Original: {orig}, 2 GPU: {two_gpu}, 1 GPU: {one_gpu}, Expected: {expected}"
            )
            all_good = False

    return all_good


def test_pytorch_lightning():
    print_section("PyTorch Lightning Check")
    try:
        import pytorch_lightning as pl

        print(f"✅ PyTorch Lightning version: {pl.__version__}")
        return True
    except ImportError:
        print("❌ PyTorch Lightning not installed")
        print("   Install with: pip install pytorch-lightning")
        return False


def test_config_loading():
    print_section("Configuration Loading Test")
    try:
        conf1 = imagenet256_autoenc()
        print("✅ Loaded imagenet256_autoenc config")
        print(f"   - Name: {conf1.name}")
        print(f"   - Image size: {conf1.img_size}")
        print(f"   - Default batch size: {conf1.batch_size}")
        print(f"   - Total samples: {conf1.total_samples}")

        conf2 = imagenet256_autoenc_latent()
        print("\n✅ Loaded imagenet256_autoenc_latent config")
        print(f"   - Name: {conf2.name}")
        print(f"   - Default batch size: {conf2.batch_size}")

        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False


def test_memory_stress(batch_size=32, img_size=256):
    """Quick memory test with dummy tensors"""
    print_section("GPU Memory Stress Test")

    if not torch.cuda.is_available():
        print("⚠️  Skipping (no GPU)")
        return True

    try:
        device = torch.device("cuda:0")
        print(f"Testing with batch_size={batch_size}, img_size={img_size}")

        # Clear cache
        torch.cuda.empty_cache()

        # Get initial memory
        memory_before = torch.cuda.memory_allocated(device) / 1024**3

        # Create dummy batch
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        # Simulate some operations
        y = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)(x)
        loss = y.mean()
        loss.backward()

        memory_after = torch.cuda.memory_allocated(device) / 1024**3
        memory_used = memory_after - memory_before

        print("✅ Test passed!")
        print(f"   Memory used: {memory_used:.2f} GB")
        print(f"   Total allocated: {memory_after:.2f} GB")

        # Cleanup
        del x, y, loss
        torch.cuda.empty_cache()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ Out of memory with batch_size={batch_size}")
            print("   Try reducing batch_size or use more gradient accumulation")
        else:
            print(f"❌ Error: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        DiffAE GPU Setup Test Script                             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # Run all tests
    results["gpu"] = check_gpu_availability()
    results["pl"] = test_pytorch_lightning()
    results["config"] = test_config_loading()
    results["batch_calc"] = test_all_configurations()
    results["memory"] = test_memory_stress()

    # Summary
    print_section("Summary")

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    if all_passed:
        print("\n" + "=" * 70)
        print("✅ All tests passed! You're ready to run training.")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Choose your configuration (1 or 2 GPUs)")
        print("2. Edit NUM_GPUS in run_imagenet256_lowgpu.py")
        print("3. Run: python run_imagenet256_lowgpu.py")
        print("\nFor a quick test run, modify the config:")
        print("  conf.total_samples = 5_000  # Test with fewer samples")
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed. Please fix the issues above.")
        print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
