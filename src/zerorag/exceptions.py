class AutoRAGError(Exception):
    """Base exception for all zerorag-related errors."""

    pass


class MissingDependencyError(AutoRAGError):
    """Raised when an optional dependency (like pypdf) is required but not installed."""

    pass
