"""Routed areas of the web map: one street graph each, built from the IBGE faces of one or more municipalities.

An area of several municipalities (the ABC Paulista) is one graph because its streets are one network across the
municipal borders: the faces of all members are joined before the corridor is drawn, so the borders add no link
of their own. Codes are IBGE's (``localidades`` API v1).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutedArea:
    id: str                         # file name of the graph and neighbourhoods (``graph/<id>.json``)
    name: str
    uf: str
    municipalities: tuple[int, ...]


AREAS: tuple[RoutedArea, ...] = (
    RoutedArea("4113700", "Londrina", "PR", (4113700,)),
    RoutedArea("4106902", "Curitiba", "PR", (4106902,)),
    RoutedArea("4205407", "Florianópolis", "SC", (4205407,)),
    RoutedArea("5300108", "Brasília", "DF", (5300108,)),
    RoutedArea("3550308", "São Paulo", "SP", (3550308,)),
    RoutedArea("abc-paulista", "ABC Paulista", "SP", (
        3547809,  # Santo André
        3548708,  # São Bernardo do Campo
        3548807,  # São Caetano do Sul
        3513801,  # Diadema
        3529401,  # Mauá
        3543303,  # Ribeirão Pires
        3544103,  # Rio Grande da Serra
    )),
)


def area(area_id: str) -> RoutedArea:
    for a in AREAS:
        if a.id == area_id:
            return a
    raise KeyError(f"unknown routed area {area_id!r}; known: {', '.join(a.id for a in AREAS)}")
