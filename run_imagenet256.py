from templates import imagenet256_autoenc, train
from templates_latent import imagenet256_autoenc_latent

# import torchvision
# torchvision.set_image_backend("PIL")
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    # Step 1: train the autoencoder
    gpus = [0, 1, 2, 3]
    conf = imagenet256_autoenc()
    train(conf, gpus=gpus)

    # Step 2: infer latents for training the latent DPM
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ["infer"]
    train(conf, gpus=gpus, mode="eval")

    # Step 3: train the latent DPM
    gpus = [0]
    conf = imagenet256_autoenc_latent()
    train(conf, gpus=gpus)

    # Step 4: unconditional sampling + FID evaluation
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ["fid(10,10)"]
    train(conf, gpus=gpus, mode="eval")
