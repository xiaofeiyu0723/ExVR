from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0


@dataclass(slots=True)
class LandmarkList:
    landmark: list[Landmark] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedLandmarkList:
    landmark: list[Landmark] = field(default_factory=list)


@dataclass(slots=True)
class Category:
    index: int
    score: float
    category_name: str = ""
    display_name: str = ""


@dataclass(slots=True)
class Classification:
    index: int
    score: float
    label: str
    display_name: str = ""


@dataclass(slots=True)
class ClassificationList:
    classification: list[Classification] = field(default_factory=list)
