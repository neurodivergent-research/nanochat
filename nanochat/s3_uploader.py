"""
Async S3 checkpoint uploader.

Uploads training checkpoints to an S3-compatible bucket in a background
daemon thread without interrupting GPU training.

Architecture:
    Training loop (rank 0)                Upload worker (daemon thread)
    ----------------------                ------------------------------
    save_checkpoint() -> disk
           |
    enqueue(step) --> Queue(maxsize=2) --> dequeue
           |                               upload files
    continue training                      write manifest (last)
    (no blocking)                          optionally delete optimizer shards
                                           log result

    When queue is full, the oldest pending job is dropped -- a newer
    checkpoint always supersedes an older one.
"""
import os
import json
import logging
import threading
import queue
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Optimizer state bytes per parameter (upper bound).
# Adam: 2 fp32 states (momentum + variance) = 8 bytes/param.
# Muon: 1 fp32 state (momentum) = 4 bytes/param.
# We use the Adam figure as a conservative upper bound since both are present.
_OPTIM_BYTES_PER_PARAM = 8
# Model bytes per parameter in bf16.
_MODEL_BYTES_PER_PARAM = 2


def estimate_checkpoint_bytes(num_params: int, world_size: int, upload_optimizer: bool) -> int:
    """
    Estimate total bytes to upload for one checkpoint.

    Model (bf16): num_params * 2
    Optimizer (all shards combined): num_params * 8  (Adam upper bound)
    Meta JSON: negligible

    Args:
        num_params: total model parameters
        world_size: number of DDP ranks (all shards uploaded by rank 0)
        upload_optimizer: whether optimizer shards are included

    Returns:
        Estimated total bytes to upload.
    """
    ...



def build_s3_prefix(
    depth: int,
    matrix_lr: float,
    total_batch_size: int,
    max_seq_len: int,
    timestamp: str,
    extra: Optional[dict[str, str]] = None,
) -> str:
    """
    Build a verbose S3 key prefix following project naming conventions.

    Args:
        depth: model depth (e.g. 20)
        matrix_lr: primary learning rate (Muon, for matrix params)
        total_batch_size: total batch size in tokens
        max_seq_len: max sequence length
        timestamp: run start time, format "YYYY-MM-DD_HH-MM-SS"
        extra: optional additional key-value pairs to include

    Returns:
        e.g. "d20_lr-0.02_bs-524288_seq-2048_2026-04-13_15-30-42"
    """
    ...


@dataclass
class S3UploadConfig:
    """Configuration for the S3 checkpoint uploader."""
    endpoint_url: str                           # S3-compatible endpoint (e.g. https://s3.us-east-1.amazonaws.com)
    bucket: str                                 # bucket name
    prefix: str                                 # key prefix (from build_s3_prefix)
    enabled: bool = True
    upload_optimizer: bool = True               # upload optimizer shards
    keep_local: int = 3                         # keep last N uploaded checkpoints on disk, delete older ones
    keep_model_on_delete: bool = True           # when deleting, preserve model + meta locally (only delete optimizer shards)
    max_retries: int = 3                        # per-file retry count
    queue_size: int = 2                         # max pending upload jobs


@dataclass
class _UploadJob:
    """Internal: a unit of work for the upload worker."""
    step: int
    checkpoint_dir: str
    upload_paths: list[str]     # local paths to upload
    s3_keys: list[str]          # corresponding S3 keys
    deletable_paths: list[str]  # paths safe to delete after upload (optimizer shards)


_SENTINEL = object()  # poison pill for clean shutdown


class S3Uploader:
    """
    Background S3 uploader for training checkpoints.

    Only rank 0 should instantiate this. Runs a single daemon thread that
    picks jobs off a bounded queue, uploads files, writes a manifest, and
    optionally cleans up local optimizer shards.

    Usage in base_train.py:
        from nanochat.s3_uploader import S3Uploader, S3UploadConfig, build_s3_prefix

        if master_process:
            prefix = build_s3_prefix(depth, matrix_lr, total_batch_size, max_seq_len, run_timestamp)
            s3_cfg = S3UploadConfig(endpoint_url=..., bucket=..., prefix=prefix)
            uploader = S3Uploader(s3_cfg, world_size=ddp_world_size)

        # after save_checkpoint():
        if master_process:
            uploader.enqueue(step, checkpoint_dir)

        # end of training:
        if master_process:
            uploader.shutdown()
    """

    def __init__(self, config: S3UploadConfig, world_size: int):
        """
        Args:
            config: S3 upload configuration
            world_size: number of DDP ranks (to know how many optimizer shards exist)
        """
        ...

    def _make_client(self):
        """
        Create and return a boto3 S3 client.

        Called inside the worker thread (boto3 clients can have thread-local state).
        Respects config.endpoint_url.
        """
        ...

    def enqueue(self, step: int, checkpoint_dir: str) -> bool:
        """
        Enqueue a checkpoint for background upload.

        Gathers file paths for the step, builds S3 keys, pushes an _UploadJob
        to the queue. If queue is full, drops the oldest pending job (newer
        checkpoints supersede older ones).

        Args:
            step: training step number
            checkpoint_dir: local directory containing checkpoint files

        Returns:
            False if uploader is disabled or shut down, True otherwise.
        """
        ...

    def _build_job(self, step: int, checkpoint_dir: str) -> _UploadJob:
        """
        Collect local file paths and compute S3 keys for a checkpoint step.

        Files per checkpoint (following checkpoint_manager.py conventions):
            model_{step:06d}.pt          -- always
            meta_{step:06d}.json         -- always
            optim_{step:06d}_rank{N}.pt  -- when config.upload_optimizer is True

        S3 keys:
            {prefix}/model_{step:06d}.pt
            {prefix}/meta_{step:06d}.json
            {prefix}/optim_{step:06d}_rank{N}.pt

        deletable_paths: optimizer shards only (when config.keep_model_on_delete).
        """
        ...

    def _worker(self):
        """
        Upload worker loop, runs in daemon thread.

        Maintains a list of completed _UploadJobs (ordered by step).
        After each successful upload:
            1. Dequeue a job (blocking). Exit on _SENTINEL.
            2. For each file, call _upload_file. Track successes.
            3. Write manifest via _upload_manifest (must be last).
            4. Append job to completed list.
            5. Call _cleanup_old_local to delete optimizer shards from
               checkpoints older than the most recent config.keep_local.
            6. Log summary (step, files uploaded, elapsed time).
        """
        ...

    def _upload_file(self, client, local_path: str, s3_key: str) -> bool:
        """
        Upload a single file to S3 with retry and exponential backoff.

        Uses multipart upload for large files (boto3 handles this via
        TransferConfig thresholds).

        Args:
            client: boto3 S3 client
            local_path: path to local file
            s3_key: destination key in the bucket

        Returns:
            True on success, False after exhausting config.max_retries.
        """
        ...

    def _upload_manifest(self, client, step: int, uploaded_files: list[dict]) -> bool:
        """
        Upload a manifest JSON as the final object for a checkpoint.

        The manifest is the atomicity marker: its presence in S3 means the
        checkpoint is complete and safe to use. Contains:
            {
                "step": 5000,
                "files": [
                    {"key": "...", "size_bytes": 12345678},
                    ...
                ]
            }

        S3 key: {prefix}/_MANIFEST_{step:06d}.json

        Returns:
            True on success.
        """
        ...

    def _cleanup_old_local(self, completed_jobs: list[_UploadJob]):
        """
        Delete local files from checkpoints that have aged out.

        Keeps the most recent config.keep_local checkpoints on disk.
        For older ones, deletes job.deletable_paths (optimizer shards
        when config.keep_model_on_delete is True, all checkpoint files
        otherwise).

        Called by the worker after each successful upload. completed_jobs
        is sorted by step (append order). Never raises -- a failed
        delete is logged and ignored.
        """
        ...

    def shutdown(self, timeout: float = 300.0):
        """
        Drain the queue and stop the worker thread.

        Sends _SENTINEL to the queue so the worker finishes its current
        job and exits cleanly. Waits up to `timeout` seconds for the
        thread to join.

        Safe to call multiple times.
        """
        ...
