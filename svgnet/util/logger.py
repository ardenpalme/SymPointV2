import os, logging
from .dist import is_main_process, master_only


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger("isbnet")
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=log_level)
    if not is_main_process():
        logger.setLevel("ERROR")
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


class SummaryWriter:
    """The tensorboardX SummaryWriter calls train.py makes, logged to wandb instead.

    Resumes the run in $WANDB_RUN_ID (ml_pipe's finetune_spv2.py creates it with the provenance);
    without one, starts a new run in $WANDB_PROJECT. Scalars are plotted against "epoch".
    wandb is imported here, not at module level: svgnet.model imports this module on the inference path.
    """

    @master_only
    def __init__(self, log_dir=None):
        import wandb

        run_id = os.environ.get("WANDB_RUN_ID")
        self._run = wandb.init(project=os.environ.get("WANDB_PROJECT"), id=run_id,
                               resume="must" if run_id else None, dir=log_dir)
        self._run.define_metric("epoch")
        self._run.define_metric("train/*", step_metric="epoch")
        self._run.define_metric("val/*", step_metric="epoch")

    @master_only
    def add_scalar(self, tag, scalar_value, global_step):
        self._run.log({tag: scalar_value, "epoch": global_step})

    @master_only
    def flush(self):
        pass  # wandb flushes on its own; train.py calls this once per epoch
