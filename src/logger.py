from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

folder_name = "logs/drqn_agent"

logger = SummaryWriter(folder_name)