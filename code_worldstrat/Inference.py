from src.train import *
import torch
import gc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["http_proxy"] = "http://127.0.0.1:1234"
os.environ["https_proxy"] = "http://127.0.0.1:1234"

default_train_command = [
    # Batch size, gpus, limits
    "python",
    "--accelerator", "gpu",
    "--devices", "1",
    "--precision", "32",
    "--seed", "431608443",
    "--data_split_seed", "386564310",
    "--batch_size", "16" ,
    "--num_workers", "4",
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
    "--root", "\dataset_entire",           # 下载数据(https://worldstrat.github.io/)放于同级别文件夹下
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
        Generating a super-resolved output
        Using the pre-trained, or any other trained model, we can easily generate super-resolved outputs by passing the low-quality images to the model.
    ----------------------------------Readme---------------------------------------------------------------------------------------

'''


if __name__ == '__main__':
    sys.argv = default_train_command
    sys.argv += ["--ourMISRmodel", "HighResNet", "--ourSharpeningmodel", "PANNet", "--use_artificial_dataset"]

    args = parse_arguments()
    set_random_seed(args)
    initialise_wandb(args)

    dataloaders = load_dataset(args)
    generate_model_backbone(args, dataloaders)
    add_gpu_augmentations(args)
    model = LitModel.load_from_checkpoint(
        checkpoint_path="   ",
        backbone = args.backbone,
        weights_only=False
    )
    print(model.hparams)
    add_callbacks(args, dataloaders)
    trainer = pl.Trainer.from_argparse_args(args)


    trainer.predict(model=model, dataloaders=[dataloaders["test"]])
    finish_wandb_logging(args)

