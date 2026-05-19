import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

from utils.utils import *
from utils.models import *

from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():

    parser = argparse.ArgumentParser()

    # ---------------- DATASET PATHS ---------------- #

    parser.add_argument(
        '--content_dir',
        type=str,
        default='../content_data',
        help='Path to content dataset'
    )

    parser.add_argument(
        '--style_dir',
        type=str,
        default='../style_data/resized/resized',
        help='Path to style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default='vgg_normalised.pth',
        help='Path to pretrained VGG weights'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment1',
        help='Experiment name'
    )

    # ---------------- IMAGE SETTINGS ---------------- #

    parser.add_argument(
        '--final_size',
        type=int,
        default=256,
        help='Final cropped image size'
    )

    parser.add_argument(
        '--content_size',
        type=int,
        default=512,
        help='Resize content image'
    )

    parser.add_argument(
        '--style_size',
        type=int,
        default=512,
        help='Resize style image'
    )

    parser.add_argument(
        '--crop',
        action='store_true',
        default=True,
        help='Use random crop'
    )

    # ---------------- TRAINING SETTINGS ---------------- #

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--lr_decay',
        type=float,
        default=5e-5,
        help='Learning rate decay'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs'
    )

    parser.add_argument(
        '--content_weight',
        type=float,
        default=1.0,
        help='Content loss weight'
    )

    parser.add_argument(
        '--style_weight',
        type=float,
        default=5.0,
        help='Style loss weight'
    )

    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='Logging interval'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=1,
        help='Checkpoint save interval'
    )

    # ---------------- RESUME TRAINING ---------------- #

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training'
    )

    parser.add_argument(
        '--decoder_path',
        type=str,
        default=None,
        help='Decoder checkpoint path'
    )

    parser.add_argument(
        '--optimizer_path',
        type=str,
        default=None,
        help='Optimizer checkpoint path'
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    # ---------------- DEVICE ---------------- #

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------- SAVE DIRECTORY ---------------- #

    save_dir = Path('experiment') / args.experiment

    save_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    # ---------------- SAVE CONFIG ---------------- #

    with open(save_dir / 'args.txt', 'w') as args_file:

        for key, value in vars(args).items():

            args_file.write(f'{key}: {value}\n')

    # ---------------- TRANSFORMS ---------------- #

    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size
    )

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size
    )

    # ---------------- DATASETS ---------------- #

    print("\nLoading datasets...")

    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    # ---------------- DATALOADERS ---------------- #

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=2
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=2
    )

    print(f"\nContent Images: {len(content_dataset)}")
    print(f"Style Images: {len(style_dataset)}")

    print(f"\nContent Batches: {len(content_dataloader)}")
    print(f"Style Batches: {len(style_dataloader)}")

    # ---------------- MODELS ---------------- #

    print("\nLoading models...")

    encoder = VGGEncoder(args.vgg).to(device)

    decoder = Decoder().to(device)

    encoder.eval()

    # ---------------- OPTIMIZER ---------------- #

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            1.0 / (1.0 + args.lr_decay * epoch)
        )
    )

    # ---------------- RESUME ---------------- #

    if args.resume:

        print("\nResuming training...")

        decoder.load_state_dict(
            torch.load(
                args.decoder_path,
                map_location=device
            )
        )

        optimizer.load_state_dict(
            torch.load(
                args.optimizer_path,
                map_location=device
            )
        )

    # ---------------- LOSS ---------------- #

    mse_loss = torch.nn.MSELoss()

    print("\nTraining Started...\n")

    # ---------------- TRAINING LOOP ---------------- #

    for epoch in range(args.epochs):

        running_loss = 0
        running_closs = 0
        running_sloss = 0

        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=min(
                len(content_dataloader),
                len(style_dataloader)
            )
        )

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)

            style_batch = style_batch.to(device)

            # ---------- ENCODE ---------- #

            c_feats = encoder(content_batch)

            s_feats = encoder(style_batch)

            # ---------- ADAIN ---------- #

            t = adaptive_instance_normalization(
                c_feats[-1],
                s_feats[-1]
            )

            # ---------- DECODE ---------- #

            g = decoder(t)

            # ---------- RE-ENCODE ---------- #

            g_feats = encoder(g)

            # ---------- CONTENT LOSS ---------- #

            loss_c = mse_loss(
                g_feats[-1],
                t
            ) * args.content_weight

            # ---------- STYLE LOSS ---------- #

            loss_s = 0

            for g_f, s_f in zip(g_feats, s_feats):

                g_mean, g_std = calc_mean_std(g_f)

                s_mean, s_std = calc_mean_std(s_f)

                loss_s += (
                    mse_loss(g_mean, s_mean)
                    + mse_loss(g_std, s_std)
                )

            loss_s *= args.style_weight

            # ---------- TOTAL LOSS ---------- #

            loss = loss_c + loss_s

            # ---------- BACKPROP ---------- #

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # ---------- STATS ---------- #

            running_loss += loss.item()

            running_closs += loss_c.item()

            running_sloss += loss_s.item()

            progress_bar.set_description(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Loss: {loss.item():.4f} | "
                f"C: {loss_c.item():.4f} | "
                f"S: {loss_s.item():.4f}"
            )

        # ---------- LR STEP ---------- #

        scheduler.step()

        # ---------- AVERAGE LOSSES ---------- #

        running_loss /= len(content_dataloader)

        running_closs /= len(content_dataloader)

        running_sloss /= len(content_dataloader)

        print(
            f"\nEpoch {epoch+1} Completed | "
            f"Loss: {running_loss:.4f} | "
            f"Content: {running_closs:.4f} | "
            f"Style: {running_sloss:.4f}"
        )

        # ---------- SAVE CHECKPOINTS ---------- #

        if (epoch + 1) % args.save_interval == 0:

            print("\nSaving checkpoints...")

            torch.save(
                decoder.state_dict(),
                save_dir / f'decoder_epoch_{epoch+1}.pth'
            )

            torch.save(
                optimizer.state_dict(),
                save_dir / f'optimizer_epoch_{epoch+1}.pth'
            )

            with torch.no_grad():

                output = torch.cat(
                    [content_batch, style_batch, g],
                    dim=0
                )

                save_image(
                    output,
                    save_dir / f'output_epoch_{epoch+1}.png',
                    nrow=args.batch_size
                )

    print("\nTraining Finished Successfully!")


if __name__ == '__main__':

    main()