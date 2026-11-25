import heapq
from typing import Callable, Optional
from puzzle.base_search import BaseSearch
from puzzle.state import State, GOAL_STATE
from puzzle.result import SearchResult


class AStar(BaseSearch):

    def __init__(self, heuristic: Optional[Callable[[State], int]] = None):
        self.custom_heuristic = heuristic

    def heuristic(self, state: State) -> int:
        if self.custom_heuristic is not None:
            return self.custom_heuristic(state)

        # distância de Manhattan de cada peça até a posição final
        total = 0
        for idx, tile in enumerate(state.tiles):
            if tile == 0:
                continue
            goal_idx = GOAL_STATE.index(tile)
            row, col = divmod(idx, 3)
            goal_row, goal_col = divmod(goal_idx, 3)
            total += abs(row - goal_row) + abs(col - goal_col)

        return total

    def search(self, initial: State) -> SearchResult:
        if initial.is_goal:
            return SearchResult(
                solution=initial,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                depth=0,
            )

        tie_breaker = 0  # evita comparar States diretamente quando f(n) empata
        frontier = [(self.heuristic(initial), tie_breaker, initial)]
        g_cost = {initial: 0}
        closed = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        while frontier:
            if len(frontier) > max_frontier_size:
                max_frontier_size = len(frontier)

            _, _, current = heapq.heappop(frontier)

            if current in closed:
                continue

            if current.is_goal:
                return SearchResult(
                    solution=current,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    depth=current.cost,
                )

            closed.add(current)
            nodes_expanded += 1

            for child in current.neighbors():
                if child in closed:
                    continue

                new_g = current.cost + 1
                if child not in g_cost or new_g < g_cost[child]:
                    g_cost[child] = new_g
                    tie_breaker += 1
                    f_cost = new_g + self.heuristic(child)
                    nodes_generated += 1
                    heapq.heappush(frontier, (f_cost, tie_breaker, child))

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            depth=0,
        )
