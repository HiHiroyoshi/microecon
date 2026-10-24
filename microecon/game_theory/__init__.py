"""ゲーム理論モジュール.

戦略型ゲーム（Normal Form Game）・ナッシュ均衡分析、Lemke-Howson アルゴリズム、
連続戦略ゲーム、展開型ゲーム（後方帰納法による部分ゲーム完全均衡）を提供する。
"""

from microecon.game_theory.continuous import ContinuousGame
from microecon.game_theory.dataclasses import (
    ContinuousGameResult,
    ExtensiveFormNode,
    MixedStrategyProfile,
    NormalFormGameResult,
    PureStrategyProfile,
    SubgamePerfectResult,
)
from microecon.game_theory.extensive import ExtensiveFormGame
from microecon.game_theory.lemke_howson import LemkeHowsonSolver
from microecon.game_theory.nash_solver import NashSolver
from microecon.game_theory.normal_form import NormalFormGame

__all__ = [
    "ContinuousGame",
    "ContinuousGameResult",
    "ExtensiveFormGame",
    "ExtensiveFormNode",
    "LemkeHowsonSolver",
    "MixedStrategyProfile",
    "NashSolver",
    "NormalFormGame",
    "NormalFormGameResult",
    "PureStrategyProfile",
    "SubgamePerfectResult",
]
