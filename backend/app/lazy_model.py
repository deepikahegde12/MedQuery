"""Thread-safe lazy loading for Hugging Face models.

Model weights are loaded on first use (or in a background warm-up thread), never
inside a request handler's import path, and a load failure is recorded instead of
crashing the app.
"""

import logging
import threading
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    UNAVAILABLE = "unavailable"


class LazyModel:
    def __init__(self, label: str, model_name: str, loader: Callable[[], Any]):
        self.label = label
        self.model_name = model_name
        self._loader = loader
        self._lock = threading.Lock()
        self._value: Any = None
        self.state = ModelState.NOT_LOADED
        self.error: str | None = None

    def get(self) -> Any | None:
        """Return the loaded object, loading it if needed; None if unavailable."""
        if self.state == ModelState.READY:
            return self._value
        if self.state == ModelState.UNAVAILABLE:
            return None
        with self._lock:
            if self.state in (ModelState.NOT_LOADED, ModelState.LOADING):
                self.state = ModelState.LOADING
                logger.info("Loading %s model '%s'...", self.label, self.model_name)
                try:
                    self._value = self._loader()
                    self.state = ModelState.READY
                    logger.info("%s model '%s' ready", self.label, self.model_name)
                except Exception as exc:  # noqa: BLE001 - missing deps, no network, bad name...
                    self.state = ModelState.UNAVAILABLE
                    self.error = f"{type(exc).__name__}: {exc}"
                    logger.warning("%s model '%s' unavailable: %s", self.label, self.model_name, self.error)
        return self._value if self.state == ModelState.READY else None

    def warm_up_in_background(self) -> None:
        threading.Thread(target=self.get, name=f"warmup-{self.label}", daemon=True).start()

    def status(self) -> dict:
        return {"model": self.model_name, "state": self.state.value, "error": self.error}
