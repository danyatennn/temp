import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.lensless import crop_roi


class Inferencer(BaseTrainer):
    """
    Inferencer (like Trainer but for inference).

    Runs the model on a dataset, optionally computes metrics, and saves the
    reconstructed region of interest as id-matched PNG files.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for inference.
            save_path (Path): directory to save reconstructions.
            metrics (dict | None): metrics for inference.
            batch_transforms (dict | None): batch transforms.
            skip_model_load (bool): if True, do not load a checkpoint.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): logs for each partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run a batch through the model, compute metrics, and save predictions.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): batch from the dataloader.
            metrics (MetricTracker | None): metric tracker.
            part (str): partition name.
        Returns:
            batch (dict): batch with model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None and "gt" in batch:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        reconstruction = crop_roi(batch["reconstruction"]).clamp(0, 1)
        for i in range(reconstruction.shape[0]):
            image = reconstruction[i].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            Image.fromarray(image).save(self.save_path / f"{batch['id'][i]}.png")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions.

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): metrics calculated on the partition.
        """
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        self.save_path.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        return {}
