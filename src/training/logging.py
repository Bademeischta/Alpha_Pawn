from torch.utils.tensorboard import SummaryWriter
import wandb


def init_logging(log_dir: str, config: dict, project: str = "chess-rl"):
    writer = SummaryWriter(log_dir)
    wandb.init(project=project, config=config)
    return writer
