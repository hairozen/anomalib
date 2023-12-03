import logging
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


class AiVadModelCheckpointCallback(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:

        # Saves the memory banks parameters
        logger.info("Saving the estimators memory banks")

        if pl_module.model.density_estimator.use_velocity_features:
            checkpoint["velocity_estimator_memory_bank"] = pl_module.model.density_estimator.velocity_estimator.memory_bank
        if pl_module.model.density_estimator.use_pose_features:
            checkpoint["pose_estimator_memory_bank"] = pl_module.model.density_estimator.pose_estimator.memory_bank
        if pl_module.model.density_estimator.use_deep_features:
            checkpoint["appearance_estimator_memory_bank"] = pl_module.model.density_estimator.appearance_estimator.memory_bank

        # Call the parent method to perform the actual saving
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

