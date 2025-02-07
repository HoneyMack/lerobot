from pathlib import Path
import argparse
import torch
import wandb

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PI0Policy with single-GPU and wandb logging.")

    parser.add_argument("--repository_id", type=str, default="gello_dataset",
                        help="Repository ID for the dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs/train/lite6_pi0",
                        help="Directory to save training outputs and checkpoints.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--training_steps", type=int, default=200000,
                        help="Total number of training steps.")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="Logging frequency in steps.")
    parser.add_argument("--wandb_project", type=str, default="pi0_training",
                        help="Wandb project name for logging.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity (username or team). Set if needed.")

    return parser.parse_args()


def main(args):
    """Main training function."""
    # Initialize wandb logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"pi0_training_{args.repository_id}",
    )

    # Create a directory to store the training checkpoint
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Set up the dataset
    fps = 50
    chunk_size = 50
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    device = torch.device("cuda")

    dataset = LeRobotDataset(
        repo_id=args.repository_id,
        delta_timestamps=delta_timestamps,
        root=Path("data/converted_datasets/lite6"),
        local_files_only=True,
        tolerance_s=0.5,
    )

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id=args.repository_id,
        root=Path("data/converted_datasets/lite6"),
        local_files_only=True,
    )
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Initialize the policy
    cfg = PI0Config(
        max_state_dim=48,
        input_features=input_features,
        output_features=output_features
    )
    policy = PI0Policy(cfg, dataset_stats=dataset.meta.stats)
    policy.train()
    policy.to(device)

    # Set up the optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v if k == "task" else v.to(device, non_blocking=True)) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log metrics to wandb
            wandb.log({"step": step, "loss": loss.item()})

            if step % args.log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

                # Save a policy checkpoint
                save_dir = output_directory / f"checkpoint_{step}"
                policy.save_pretrained(save_dir)

            step += 1
            if step >= args.training_steps:
                done = True
                break

    # Save the latest policy checkpoint
    save_dir = output_directory / "checkpoint_latest"
    policy.save_pretrained(save_dir)

    # Finish wandb logging
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)