"""産業組織論モジュール（独占・価格差別・寡占・市場構造比較）."""

from microecon.industrial_organization.bertrand import DifferentiatedBertrand, HomogeneousBertrand
from microecon.industrial_organization.comparison import compare_market_structures
from microecon.industrial_organization.cournot import CournotDuopoly
from microecon.industrial_organization.dataclasses import (
    MarketStructureComparison,
    MarketStructureRow,
    MonopolyResult,
    OligopolyResult,
    ThirdDegreeResult,
)
from microecon.industrial_organization.monopoly import Monopoly, ThirdDegreePriceDiscrimination
from microecon.industrial_organization.plotter import IOPlotter
from microecon.industrial_organization.stackelberg import StackelbergDuopoly

__all__ = [
    "CournotDuopoly",
    "DifferentiatedBertrand",
    "HomogeneousBertrand",
    "IOPlotter",
    "MarketStructureComparison",
    "MarketStructureRow",
    "Monopoly",
    "MonopolyResult",
    "OligopolyResult",
    "StackelbergDuopoly",
    "ThirdDegreePriceDiscrimination",
    "ThirdDegreeResult",
    "compare_market_structures",
]
