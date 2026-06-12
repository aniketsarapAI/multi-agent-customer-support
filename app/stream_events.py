import queue
import threading
from langchain_core.callbacks import BaseCallbackHandler


class TokenCollector(BaseCallbackHandler):
    """LangChain callback that collects tokens into a thread-safe buffer.

    Designed to be passed via `config["callbacks"]` when calling
    ``app.stream()``. Tokens emitted by any LLM call with ``streaming=True``
    inside the graph are appended to an internal queue so the UI can poll
    progressively.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._done = threading.Event()

    def on_llm_new_token(self, token: str, **kwargs):
        self._queue.put(token)

    def mark_done(self):
        self._done.set()

    def iter_tokens(self, timeout: float = 0.05):
        """Generator that yields tokens as they arrive.

        Blocks up to *timeout* seconds between polls. Stops after
        ``mark_done()`` has been called and the queue is empty.
        """
        while not self._done.is_set() or not self._queue.empty():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue
