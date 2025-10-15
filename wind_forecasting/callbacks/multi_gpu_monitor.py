"""
Multi-GPU monitoring callback for PyTorch Lightning.

This callback monitors all GPUs in DDP training and logs metrics to WandB,
addressing the limitation where WandB's default system monitor only tracks GPU 0.
"""

import logging
import subprocess
from typing import Optional
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class MultiGPUMonitor(Callback):
    """
    Monitor all GPUs in DDP training and log comprehensive metrics to WandB.

    This callback queries nvidia-smi on each training batch to collect GPU metrics
    from all available GPUs, not just the one visible to rank 0.

    Parameters
    ----------
    log_interval : int, default=10
        Log GPU metrics every N training batches
    timeout_seconds : float, default=2.0
        Timeout for nvidia-smi subprocess call
    """

    def __init__(self, log_interval: int = 10, timeout_seconds: float = 2.0):
        super().__init__()
        self.log_interval = log_interval
        self.timeout_seconds = timeout_seconds
        self._warned_once = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int
    ) -> None:
        """
        Query and log GPU metrics at specified intervals.

        Only logs from rank 0 to avoid duplicate logging in DDP.
        """
        # Only log from rank 0 to avoid duplicate logging
        if not trainer.is_global_zero:
            return

        # Log at specified intervals
        if batch_idx % self.log_interval != 0:
            return

        try:
            # Create environment without CUDA_VISIBLE_DEVICES to query ALL physical GPUs
            # SLURM's gpu-bind sets CUDA_VISIBLE_DEVICES per rank, limiting nvidia-smi to one GPU
            import os
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)  # Remove CUDA restriction

            # Query nvidia-smi for comprehensive GPU metrics from ALL GPUs
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,fan.speed',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=env  # Use modified environment
            )

            if result.returncode != 0:
                if not self._warned_once:
                    logger.warning(
                        f"nvidia-smi returned non-zero exit code: {result.returncode}. "
                        f"Multi-GPU monitoring may not work correctly."
                    )
                    self._warned_once = True
                return

            # Parse and log metrics for each GPU
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                try:
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) < 7:
                        continue

                    # Extract metrics
                    gpu_idx = parts[0]
                    gpu_util = self._safe_float(parts[1])
                    mem_util = self._safe_float(parts[2])
                    mem_used = self._safe_float(parts[3])
                    mem_total = self._safe_float(parts[4])
                    power = self._safe_float(parts[5])
                    temp = self._safe_float(parts[6])

                    # Fan speed is optional (might be N/A on some GPUs)
                    fan_speed = self._safe_float(parts[7]) if len(parts) >= 8 else None

                    # Calculate memory percentage if total is available
                    mem_percent = None
                    if mem_total is not None and mem_total > 0 and mem_used is not None:
                        mem_percent = (mem_used / mem_total) * 100.0

                    # Log comprehensive metrics
                    metrics = {
                        f'system/gpu.{gpu_idx}.utilization': gpu_util,
                        f'system/gpu.{gpu_idx}.memory_utilization': mem_util,
                        f'system/gpu.{gpu_idx}.memory_used_mb': mem_used,
                        f'system/gpu.{gpu_idx}.memory_total_mb': mem_total,
                        f'system/gpu.{gpu_idx}.power_watts': power,
                        f'system/gpu.{gpu_idx}.temperature_c': temp,
                    }

                    # Add optional metrics if available
                    if mem_percent is not None:
                        metrics[f'system/gpu.{gpu_idx}.memory_percent'] = mem_percent

                    if fan_speed is not None:
                        metrics[f'system/gpu.{gpu_idx}.fan_speed_percent'] = fan_speed

                    # Filter out None values
                    metrics = {k: v for k, v in metrics.items() if v is not None}

                    # Log to WandB using Lightning's logger interface
                    # This ensures metrics are tied to the current global_step
                    if trainer.logger is not None:
                        trainer.logger.log_metrics(metrics, step=trainer.global_step)

                except (ValueError, IndexError) as e:
                    if not self._warned_once:
                        logger.warning(f"Failed to parse GPU metrics line: {line}. Error: {e}")
                        self._warned_once = True

        except subprocess.TimeoutExpired:
            if not self._warned_once:
                logger.warning(
                    f"nvidia-smi query timed out after {self.timeout_seconds}s. "
                    f"Increase timeout_seconds or reduce log_interval."
                )
                self._warned_once = True
        except FileNotFoundError:
            if not self._warned_once:
                logger.warning(
                    "nvidia-smi not found. Multi-GPU monitoring requires NVIDIA drivers and nvidia-smi."
                )
                self._warned_once = True
        except Exception as e:
            if not self._warned_once:
                logger.warning(f"Unexpected error in multi-GPU monitoring: {e}")
                self._warned_once = True

    @staticmethod
    def _safe_float(value: str) -> Optional[float]:
        """
        Safely convert string to float, returning None for invalid values.

        Parameters
        ----------
        value : str
            String value to convert

        Returns
        -------
        float or None
            Converted float or None if conversion fails
        """
        try:
            value = value.strip()
            if value.lower() in ['n/a', '', 'not supported']:
                return None
            return float(value)
        except (ValueError, AttributeError):
            return None
