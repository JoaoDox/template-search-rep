from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult

DEFAULT_DEPTH_LIMIT = 50


class DFS(BaseSearch):

    def __init__(self, depth_limit: int = DEFAULT_DEPTH_LIMIT):
        self.depth_limit = depth_limit

    def search(self, initial: State) -> SearchResult:
        if initial.is_goal:
            return SearchResult(
                solution=initial,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                depth=0,
            )

        # pilha guarda (estado, profundidade atual)
        frontier = [(initial, 0)]
        # guarda a menor profundidade em que cada estado já foi alcançado
        visited_depth = {initial: 0}

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        while frontier:
            if len(frontier) > max_frontier_size:
                max_frontier_size = len(frontier)

            current, depth = frontier.pop()

            if current.is_goal:
                return SearchResult(
                    solution=current,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    depth=depth,
                )

            if depth >= self.depth_limit:
                continue

            nodes_expanded += 1

            for child in reversed(current.neighbors()):
                next_depth = depth + 1
                if child not in visited_depth or next_depth < visited_depth[child]:
                    visited_depth[child] = next_depth
                    nodes_generated += 1
                    frontier.append((child, next_depth))

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            depth=0,
        )
