"""2消費者・2財の純粋交換経済（Edgeworth Box）."""

from __future__ import annotations

from typing import ClassVar

import sympy as sp

from microecon.general_equilibrium.arrow_debreu import ArrowDebreuEconomy
from microecon.general_equilibrium.consumer import BaseConsumer
from microecon.general_equilibrium.dataclasses import (
    CompetitiveEquilibriumResult,
    EdgeworthBoxResult,
)


class EdgeworthBox:
    """2消費者・2財の純粋交換経済モデル.

    消費者 A の財1・財2の消費量をシンボル :attr:`x1A`・:attr:`x2A` で表し、
    消費者 B の消費量は総初期保有量 ``(E1, E2)`` を用いて
    ``(E1 - x1A, E2 - x2A)`` と表す。

    Raises:
        ValueError: 消費者名が同一の場合、いずれかの消費者の財の数が 2 でない場合、
            または総初期保有量が 0 の財が存在する場合。
    """

    x1A: ClassVar[sp.Symbol] = sp.Symbol("x1A", positive=True)
    x2A: ClassVar[sp.Symbol] = sp.Symbol("x2A", positive=True)

    def __init__(self, consumer_a: BaseConsumer, consumer_b: BaseConsumer) -> None:
        if consumer_a.name == consumer_b.name:
            raise ValueError("Consumer names must be unique.")
        if consumer_a.n_goods != 2 or consumer_b.n_goods != 2:
            raise ValueError(
                "EdgeworthBox requires both consumers to have exactly 2 goods, got "
                f"{consumer_a.n_goods} and {consumer_b.n_goods}."
            )
        self.consumer_a = consumer_a
        self.consumer_b = consumer_b
        self._economy = ArrowDebreuEconomy((consumer_a, consumer_b))

    @property
    def total_endowment(self) -> tuple[float, float]:
        """総初期保有量 ``(E1, E2)``."""
        e1, e2 = self._economy.total_endowment
        return (e1, e2)

    def get_contract_curve(self) -> sp.Expr:
        """契約曲線 ``x2A = f(x1A)`` の右辺を返す.

        ``MRS_A(x1A, x2A) = MRS_B(E1 - x1A, E2 - x2A)`` を ``x2A`` について解く。
        両消費者が同一の効用変数シンボルを共有し得るため、代入は各消費者の
        MRS 式に対して個別かつ同時に（``simultaneous=True``）行う。

        Raises:
            NotImplementedError: ``x1A = E1 / 2`` を代入したとき箱の内部
                ``0 < x2A < E2`` に入る実数解がちょうど1つでない場合。
        """
        e1, e2 = (sp.Rational(str(e)) for e in self.total_endowment)
        vars_a = self.consumer_a.utility_function.utility_variables
        vars_b = self.consumer_b.utility_function.utility_variables

        mrs_a = self.consumer_a.get_mrs_expression(0, 1).subs(
            {vars_a[0]: self.x1A, vars_a[1]: self.x2A}, simultaneous=True
        )
        mrs_b = self.consumer_b.get_mrs_expression(0, 1).subs(
            {vars_b[0]: e1 - self.x1A, vars_b[1]: e2 - self.x2A}, simultaneous=True
        )

        try:
            solutions = sp.solve(sp.Eq(mrs_a, mrs_b), self.x2A)
        except NotImplementedError:
            solutions = []

        interior = [sol for sol in solutions if self._is_interior(sol, e1, e2)]
        if len(interior) != 1:
            raise NotImplementedError(
                "Contract curve does not have a unique interior algebraic solution."
            )
        return sp.simplify(interior[0])

    def solve_competitive_equilibrium(
        self, numeraire_idx: int = 0
    ) -> CompetitiveEquilibriumResult:
        """基準財 ``numeraire_idx`` の価格を 1.0 に固定した競争均衡を求める."""
        return self._economy.solve(numeraire_idx=numeraire_idx)

    def analyze(self, numeraire_idx: int = 0) -> EdgeworthBoxResult:
        """契約曲線と競争均衡をまとめて算出する."""
        return EdgeworthBoxResult(
            total_endowment=self.total_endowment,
            contract_curve_expression=self.get_contract_curve(),
            competitive_equilibrium=self.solve_competitive_equilibrium(numeraire_idx),
        )

    def _is_interior(self, solution: sp.Expr, e1: sp.Expr, e2: sp.Expr) -> bool:
        value = solution.subs(self.x1A, e1 / 2)
        if value.free_symbols:
            return False
        try:
            numeric = complex(sp.N(value))
        except TypeError:
            return False
        return abs(numeric.imag) <= 1e-12 and 0.0 < numeric.real < float(e2)
