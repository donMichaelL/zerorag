class ZeroRAGError(Exception):
    """Base exception for all zerorag-related errors."""

    pass


class MissingDependencyError(ZeroRAGError):
    """Raised when an optional dependency (like pypdf) is required but not installed."""

    pass
