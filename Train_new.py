# run_experiments.py
import subprocess
import os

# 设置环境变量（和你原来保持一致）
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 基础命令
default_train_command = [
    "python", "/home/tongyufei/SatFusion/Code_WWW/train.py",   # 直接调用 train.py
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

# 定义不同实验配置

# 检验 SatFusion 要比 pansharpening，在 wald 协议下，不差（如果能好最好）
runs = [
    default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
                             "128", "--use_artificial_dataset", "--temporal_jitter", "1"],
    default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
                             "128", "--use_artificial_dataset", "--temporal_jitter", "2"],
    default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
                             "128", "--use_artificial_dataset", "--temporal_jitter", "3"],
    default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
                             "128", "--use_artificial_dataset", "--temporal_jitter", "4"],
    default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
                             "128", "--use_artificial_dataset", "--temporal_jitter", "5"],
]

# 3
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "1"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "2"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "3"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "4"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "5"],

# 4
# default_train_command + ["--ourMISRmodel", "SRCNN", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "1"],
# default_train_command + ["--ourMISRmodel", "SRCNN", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "2"],
# default_train_command + ["--ourMISRmodel", "SRCNN", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "3"],
# default_train_command + ["--ourMISRmodel", "SRCNN", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "4"],
# default_train_command + ["--ourMISRmodel", "SRCNN", "--ourSharpeningmodel", "PANNet", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "5"],

# 6
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "1"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "2"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "3"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "4"],
# default_train_command + ["--ourMISRmodel", "None", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "5"],

# 7
# default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "1"],
# default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "2"],
# default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "3"],
# default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "4"],
# default_train_command + ["--ourMISRmodel", "TRNet", "--ourSharpeningmodel", "PSIT", "--hidden_channels",
#                          "128", "--use_artificial_dataset", "--temporal_jitter", "5"],




# 顺序执行
for i, run in enumerate(runs):
    print(f"\n========== Running experiment {i} ==========\n")
    subprocess.run(run)   # 每个 run 是独立进程，结束后自动释放 /dev/shm
