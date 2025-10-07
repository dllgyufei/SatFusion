# run_experiments.py
import subprocess
import os

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '*'
os.environ["http_proxy"] = "***********"
os.environ["https_proxy"] = "***********"

# 基础命令
default_train_command = [
    "python", "/train.py",   # 调用 train.py
    "--accelerator", "gpu",
    "--devices", "1",
    "--precision", "16",
    "--seed", "122938034",
    "--data_split_seed", "386564310",
    "--batch_size", "16",
    "--num_workers", "8",
    "--max_epochs", "20",

    # Model/Hyperparameters
    "--model", "ourframework",
    "--hidden_channels", "128",
    "--zoom_factor", "2",
    "--shift_px", "2",
    "--shift_mode", "lanczos",
    "--shift_step", "0.5",
    "--learning_rate", "1e-4",
    "--use_reference_frame",

    # Data
    "--dataset", "JIF",
    "--root", "/home/tongyufei/SatFusion/dateset_entire",
    "--input_size", "160", "160",
    "--output_size", "500", "500",
    "--chip_size", "50", "50",
    "--lr_bands_to_use", "true_color",
    "--revisits", "8",

    # loss
    "--w_mse", "0.3",
    "--w_mae", "0.3",
    "--w_ssim", "0.2",
    "--w_sam", "0.2",

    # Training, validation, test splits
    "--list_of_aois", "pretrained_model/final_split.csv",
]

''' 
    ----------------------------------Readme---------------------------------------------------------------------------------------
        # use artificial dataset
        + ["--temporal_noise","0.30"]
        + ["--temporal_jitter","0.15"]
        + ["--use_artificial_dataset"]
    ----------------------------------Readme---------------------------------------------------------------------------------------
'''

runs = [
    # # do not set below params both None
    # ourMISRmodel other choice: None,HighResNet,RAMS,TRNet
    # ourSharpeningmodel other choice: None,PNN,PANNet,PSIT(INNformer is PSIT here)
    default_train_command + ["--ourMISRmodel", "SRCNN", "--ourSharpeningmodel", "Pan_Mamba"], 
]

# 顺序执行
for i, run in enumerate(runs):
    print(f"\n========== Running experiment {i} ==========\n")
    subprocess.run(run)   # 每个 run 是独立进程，结束后自动释放 /dev/shm
