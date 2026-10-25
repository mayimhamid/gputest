"""Asynchronous checkpoint writer.

The solver thread hands over a host copy of the state and continues; a
background thread writes it to disk. A bounded queue gives back-pressure: if
the disk is slower than the checkpoint rate, the solver waits instead of
filling RAM with pending snapshots.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import threading
import time
from typing import Optional

import numpy as np


class CheckpointWriter:
    def __init__(self, directory: str, keep: int = 3):
        self.directory = directory
        self.keep = keep
        os.makedirs(directory, exist_ok=True)
        self._q: "queue.Queue" = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._run, name="flowtrace-checkpoint", daemon=True)
        self._thread.start()
        self.written: list = []
        self.last_error: Optional[str] = None
        self.bytes_written = 0
        self.seconds = 0.0

    def submit(self, tag: str, state: dict, meta: dict) -> None:
        self._q.put((tag, state, meta))

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            tag, state, meta = item
            t0 = time.perf_counter()
            path = os.path.join(self.directory, f"{tag}.npz")
            tmp = path + ".tmp.npz"
            try:
                arrays = {k: np.asarray(v) for k, v in state.items()}
                arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
                np.savez(tmp, **arrays)
                os.replace(tmp, path)
                self.written.append(path)
                self.bytes_written += os.path.getsize(path)
                self._prune()
            except Exception as exc:  # never kill the solver because the disk is full
                self.last_error = str(exc)
            finally:
                self.seconds += time.perf_counter() - t0
                self._q.task_done()

    def _prune(self):
        files = sorted(glob.glob(os.path.join(self.directory, "*.npz")), key=os.path.getmtime)
        for f in files[:-self.keep]:
            try:
                os.remove(f)
            except OSError:
                pass

    def flush(self):
        self._q.join()

    def close(self):
        self.flush()
        self._q.put(None)
        self._thread.join(timeout=10)


def load_checkpoint(path: str):
    with np.load(path) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        state = {k: z[k] for k in z.files if k != "__meta__"}
    for k in ("t", "step", "dt_prev"):
        if k in state:
            state[k] = state[k].item()
    return state, meta
