import os
import lmdb
import argparse
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as trans_fn
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def resize_and_convert(img, size, resample=Image.LANCZOS, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    return buffer.getvalue()


def write_lmdb(dataset, out_path, size=256):
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    total = 0

    with lmdb.open(out_path, map_size=1024**4, readahead=False) as env:
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            img_bytes = sample["image_resized"]
            label = sample["label"]

            key = f"{size}-{str(idx).zfill(8)}".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, img_bytes)
                txn.put(
                    f"label-{str(idx).zfill(8)}".encode("utf-8"),
                    str(label).encode("utf-8"),
                )

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize and convert ImageNet subset to LMDB"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split: train or validation"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        required=True,
        help="Class IDs (integers) to include, e.g. --classes 0 1",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Target image size (default 256)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/evafs/groups/ganzha_23/mgalkowski/masters/inp_exp/src/dataset/cache",
        help="HF cache directory",
    )
    parser.add_argument(
        "--out_dir", type=str, default="datasets", help="Output directory for LMDB"
    )
    args = parser.parse_args()

    # load dataset
    dataset = load_dataset(
        "imagenet-1k",
        split=args.split,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        token=os.environ["HF_TOKEN"],
    )

    # filter only selected classes
    dataset = dataset.filter(lambda ex: ex["label"] in args.classes)

    # resize + convert to bytes with .map (can parallelize)
    def process_example(example):
        img = example["image"].convert("RGB")
        example["image_resized"] = resize_and_convert(img, args.size)
        return example

    dataset = dataset.map(process_example, num_proc=os.cpu_count())

    # output LMDB
    out_path = f"{args.out_dir}/imagenet_subset_{len(args.classes)}cls_{args.split}_{args.size}.lmdb"
    write_lmdb(dataset, out_path, size=args.size)

    print(f"✅ Finished writing LMDB: {out_path}")
