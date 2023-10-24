class Singleton:
    """Base class for singleton classes."""

    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance


class GlobType(Singleton):
    """Singleton class representing a glob pattern."""


# Singleton object representing a glob pattern.
Glob = GlobType()
