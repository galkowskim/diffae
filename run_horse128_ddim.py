from templates import horse128_ddpm
from experiment import train

if __name__ == "__main__":
    gpus = [0, 1, 2, 3]
    conf = horse128_ddpm()
    train(conf, gpus=gpus)

    gpus = [0, 1, 2, 3]
    conf.eval_programs = ["fid10"]
    train(conf, gpus=gpus, mode="eval")
