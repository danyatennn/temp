import argparse

import gdown

from src.utils.io_utils import ROOT_PATH

CHECKPOINTS = {
    "unrolled_admm20": "PUT_GDRIVE_FILE_ID_HERE",
    "modular_pre_post": "PUT_GDRIVE_FILE_ID_HERE",
    "modular_pre": "PUT_GDRIVE_FILE_ID_HERE",
    "modular_post": "PUT_GDRIVE_FILE_ID_HERE",
}


def download(name, file_id):
    """
    Download a single checkpoint into saved/<name>/model_best.pth.
    """
    save_dir = ROOT_PATH / "saved" / name
    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / "model_best.pth"
    gdown.download(id=file_id, output=str(output), quiet=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints.")
    parser.add_argument(
        "-m",
        "--model",
        default="all",
        choices=["all"] + list(CHECKPOINTS.keys()),
        help="Which checkpoint to download.",
    )
    args = parser.parse_args()

    names = list(CHECKPOINTS.keys()) if args.model == "all" else [args.model]
    for name in names:
        file_id = CHECKPOINTS[name]
        if file_id == "PUT_GDRIVE_FILE_ID_HERE":
            print(f"Skipping '{name}': set its Google Drive file id in CHECKPOINTS.")
            continue
        print(f"Downloading checkpoint for '{name}'...")
        download(name, file_id)


if __name__ == "__main__":
    main()
