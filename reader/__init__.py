"""Reader package.

Heavy training/evaluation dependencies are imported only by the commands that need
them, so production inference does not require the full datasets toolchain.
"""

__all__ = ["ReaderPredictor"]


def __getattr__(name):
    if name == "ReaderPredictor":
        from reader.predict import ReaderPredictor

        return ReaderPredictor
    raise AttributeError(name)
