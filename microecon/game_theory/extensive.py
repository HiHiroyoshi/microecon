"""完備情報動的ゲーム（展開型ゲーム）と後方帰納法による部分ゲーム完全均衡."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from microecon.game_theory.dataclasses import ExtensiveFormNode, SubgamePerfectResult

_SUPPORTED_TIE_BREAKING_RULES = ("first",)
_TIE_TOL = 1e-9


class ExtensiveFormGame:
    """完備情報の展開型ゲーム（ゲーム木）.

    Args:
        root_node_id: ルートノードのID。
        nodes: ノードID -> :class:`ExtensiveFormNode` のマッピング。
        num_players: プレイヤー数。

    Raises:
        ValueError: ``nodes`` が厳密な木構造を成さない場合（ルート不在、
            親を複数持つノード・循環・孤立ノードの存在、プレイヤー
            インデックス範囲外、終端・非終端ノードの属性不整合など）。
    """

    def __init__(
        self,
        root_node_id: str,
        nodes: Mapping[str, ExtensiveFormNode],
        num_players: int = 2,
    ) -> None:
        if num_players < 1:
            raise ValueError(f"num_players must be at least 1, got {num_players}.")
        self.root_node_id = root_node_id
        self.nodes: Mapping[str, ExtensiveFormNode] = dict(nodes)
        self.num_players = num_players
        self._validate()

    def _validate(self) -> None:
        if self.root_node_id not in self.nodes:
            raise ValueError(f"Root node '{self.root_node_id}' is not in nodes.")

        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise ValueError(
                    f"Mapping key '{node_id}' does not match node_id '{node.node_id}'."
                )
            if node.is_terminal:
                self._validate_terminal(node)
            else:
                self._validate_decision(node)

        parent_counts = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for child_id in (node.children or {}).values():
                if child_id not in self.nodes:
                    raise ValueError(
                        f"Node '{node.node_id}' references unknown child '{child_id}'."
                    )
                parent_counts[child_id] += 1

        if parent_counts[self.root_node_id] != 0:
            raise ValueError(f"Root node '{self.root_node_id}' must not have a parent.")
        for node_id, count in parent_counts.items():
            if node_id != self.root_node_id and count != 1:
                raise ValueError(
                    f"Node '{node_id}' must have exactly one parent, found {count}."
                )

        reachable = {self.root_node_id}
        queue = deque([self.root_node_id])
        while queue:
            node = self.nodes[queue.popleft()]
            for child_id in (node.children or {}).values():
                if child_id not in reachable:
                    reachable.add(child_id)
                    queue.append(child_id)
        unreachable = set(self.nodes) - reachable
        if unreachable:
            raise ValueError(
                f"Nodes unreachable from root '{self.root_node_id}': {sorted(unreachable)}."
            )

    def _validate_terminal(self, node: ExtensiveFormNode) -> None:
        if node.payoffs is None or len(node.payoffs) != self.num_players:
            raise ValueError(
                f"Terminal node '{node.node_id}' must have payoffs of length "
                f"{self.num_players}."
            )
        if node.children:
            raise ValueError(f"Terminal node '{node.node_id}' must not have children.")
        if node.actions:
            raise ValueError(f"Terminal node '{node.node_id}' must not have actions.")

    def _validate_decision(self, node: ExtensiveFormNode) -> None:
        assert node.player_idx is not None
        if not 0 <= node.player_idx < self.num_players:
            raise ValueError(
                f"Node '{node.node_id}' has player_idx {node.player_idx} outside "
                f"[0, {self.num_players})."
            )
        if not node.actions:
            raise ValueError(f"Decision node '{node.node_id}' must have actions.")
        if len(set(node.actions)) != len(node.actions):
            raise ValueError(f"Decision node '{node.node_id}' has duplicate actions.")
        if node.children is None or set(node.children) != set(node.actions):
            raise ValueError(
                f"Decision node '{node.node_id}' must have children keyed exactly by "
                "its actions."
            )
        if node.payoffs is not None:
            raise ValueError(f"Decision node '{node.node_id}' must not have payoffs.")

    def solve_spne(self, tie_breaking_rule: str = "first") -> SubgamePerfectResult:
        """後方帰納法により部分ゲーム完全ナッシュ均衡 (SPNE) を導出する.

        均衡経路外を含む全意思決定ノードでの行動を定めた完全な戦略と、
        均衡経路上で実現する利得を返す。

        Args:
            tie_breaking_rule: 無差別な行動が複数ある場合の選択規則。
                ``"first"`` は ``actions`` タプルで最も前にある行動を選ぶ。

        Raises:
            ValueError: 未対応の ``tie_breaking_rule`` が指定された場合。
        """
        if tie_breaking_rule not in _SUPPORTED_TIE_BREAKING_RULES:
            raise ValueError(f"Unsupported tie_breaking_rule '{tie_breaking_rule}'")

        strategies: dict[int, dict[str, str]] = {p: {} for p in range(self.num_players)}
        values: dict[str, tuple[float, ...]] = {}

        # 再帰上限を避けるため、明示的なスタックによる後順走査で葉から評価する。
        stack: list[tuple[str, bool]] = [(self.root_node_id, False)]
        while stack:
            node_id, children_done = stack.pop()
            node = self.nodes[node_id]
            if node.is_terminal:
                assert node.payoffs is not None
                values[node_id] = node.payoffs
                continue
            assert node.children is not None and node.player_idx is not None
            if not children_done:
                stack.append((node_id, True))
                stack.extend((child_id, False) for child_id in node.children.values())
                continue

            mover = node.player_idx
            best_action = node.actions[0]
            best_value = values[node.children[best_action]]
            for action in node.actions[1:]:
                candidate = values[node.children[action]]
                if candidate[mover] > best_value[mover] + _TIE_TOL:
                    best_action, best_value = action, candidate
            strategies[mover][node_id] = best_action
            values[node_id] = best_value

        return SubgamePerfectResult(
            equilibrium_strategies=strategies,
            payoffs=values[self.root_node_id],
        )
