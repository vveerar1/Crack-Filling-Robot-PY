"""Lightweight sectioned logger for the planners (OnlineSCC / SCC).

A single module-level ``LOG`` singleton, written to the console and/or a rotating
file with nice section rendering. **Disabled by default** -- every method is a cheap
no-op until ``LOG.enable(...)`` is called, so library callers (tests, the
automated gates) are completely unaffected and the planner stays viz/log-free.

Enable it from the CLI (``OnlineSCC.py`` does this by default; ``-q`` -> console off,
file still saved) or programmatically. File rotation: a new run writes
``logs/<name>.log``; if that already exists, the OLD file is renamed to
``logs/<name>-<n>.log`` first (lowest free n), so the latest run is always
``<name>.log`` and history is preserved.
"""

import os

_W = 70                          # section bar width
_MAJOR = "=" * _W
_MINOR = "-" * _W


class _Log:
    def __init__(self):
        self._on = False
        self._console = True
        self._fh = None
        self._path = None

    @property
    def enabled(self):
        return self._on

    @property
    def path(self):
        return self._path

    def enable(self, console=True, logfile=None):
        """Turn logging on. ``console`` -> also print to stdout. ``logfile`` -> append
        to that path (opened for writing; caller usually gets it from ``rotate_path``)."""
        self._on = True
        self._console = console
        if logfile is not None:
            d = os.path.dirname(logfile)
            if d:
                os.makedirs(d, exist_ok=True)
            self._fh = open(logfile, "w", encoding="utf-8")
            self._path = logfile

    def disable(self):
        self._on = False
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    # --- emit primitives (all no-op when disabled) ---
    def _emit(self, text=""):
        if not self._on:
            return
        if self._console:
            print(text, flush=True)
        if self._fh is not None:
            self._fh.write(text + "\n")
            self._fh.flush()

    def section(self, title):
        """Major section: blank line, '=' bar, title, '=' bar."""
        if not self._on:
            return
        self._emit("")
        self._emit(_MAJOR)
        self._emit("  " + title)
        self._emit(_MAJOR)

    def sub(self, title):
        """Minor section (e.g. per-iteration): blank line, '-' bar, title, '-' bar."""
        if not self._on:
            return
        self._emit("")
        self._emit(_MINOR)
        self._emit("  " + title)
        self._emit(_MINOR)

    def kv(self, label, value):
        """Aligned key/value line."""
        if not self._on:
            return
        self._emit(f"  {str(label):<22}{value}")

    def info(self, msg=""):
        if not self._on:
            return
        self._emit(msg)

    def exception(self, title):
        """Log a STOPPED section with the active exception's full traceback, so a
        crashed or interrupted run is visible in the log file afterwards. Call from
        an ``except`` block. No-op when logging is disabled (gates/tests unaffected)."""
        if not self._on:
            return
        import traceback
        self._emit("")
        self._emit(_MAJOR)
        self._emit("  STOPPED -- " + title)
        self._emit(_MAJOR)
        self._emit(traceback.format_exc().rstrip())


LOG = _Log()


def rotate_path(folder, name, keep=5):
    """Return ``folder/name.log`` for a fresh run. If it exists, logrotate-style shift
    the history: drop ``name-<keep>.log`` (oldest), ``name-<k>`` -> ``name-<k+1>``, and
    ``name.log`` -> ``name-1.log``. So ``name-1`` is always the most recent old run and
    at most ``keep`` (=5) numbered backups are retained."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".log")
    if os.path.exists(path):
        oldest = os.path.join(folder, f"{name}-{keep}.log")
        if os.path.exists(oldest):
            os.remove(oldest)
        for n in range(keep - 1, 0, -1):
            src = os.path.join(folder, f"{name}-{n}.log")
            if os.path.exists(src):
                os.replace(src, os.path.join(folder, f"{name}-{n + 1}.log"))
        os.replace(path, os.path.join(folder, f"{name}-1.log"))
    return path
