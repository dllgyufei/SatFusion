# train.py
from src.train import cli_main
import torch

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    cli_main()   # 直接运行，参数由 sys.argv 提供
