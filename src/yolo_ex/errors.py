"""Application exceptions."""


class ExportError(Exception):
    """Base error for export operations."""


class ExportValidationError(ExportError):
    """Raised when CLI input or export parameters are invalid."""


class ExportExecutionError(ExportError):
    """Raised when the export backend fails."""
