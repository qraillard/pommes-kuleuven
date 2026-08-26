"""Retrofit: transfer of installed capacity between technologies."""

from pommes.model.retrofit.combined import add_retrofit_combined
from pommes.model.retrofit.conversion import add_retrofit_conversion
from pommes.model.retrofit.storage import add_retrofit_storage
from pommes.model.retrofit.transport import add_retrofit_transport

__all__ = [
    "add_retrofit_combined",
    "add_retrofit_conversion",
    "add_retrofit_storage",
    "add_retrofit_transport",
]
