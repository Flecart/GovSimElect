from datetime import datetime

from simulation.utils import ModelWandbWrapper

from ..memory.associative_memory import AssociativeMemory
from .component import Component


class RetrieveComponent(Component):
    """
    Retrieve ranks private memories using recency and importance only.
    - Recency: 0.99**i over reverse file order, so newer entries score higher.
    - Importance: LLM-generated score normalized from the stored 1-10 range.
    """

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        associative_memory: AssociativeMemory,
    ):
        super().__init__(model, model_framework)
        self.associative_memory = associative_memory

        self.weights = {
            "recency": 0.5,
            "importance": 3,
        }
        self.recency_decay_param = 0.99

    def _recency_retrieval(
        self, nodes: list[tuple[datetime, str, float, bool]]
    ) -> dict[int, float]:
        """
        Calculate the recency retrieval scores for a list of nodes.

        Args:
            nodes (list[tuple[datetime, str, float, bool]]): The list of nodes to
                calculate recency retrieval scores for.

        Returns:
            dict[int, float]: A dictionary mapping node indexes to their recency
                retrieval scores.
        """
        result = dict()
        for i, node_index in enumerate(range(len(nodes) - 1, -1, -1)):
            result[node_index] = self.recency_decay_param**i
        return result

    def _importance_retrieval(
        self, nodes: list[tuple[datetime, str, float, bool]]
    ) -> dict[int, float]:
        """
        Retrieve the importance scores for a list of nodes and normalize them.

        Args:
            nodes (list[tuple[datetime, str, float, bool]]): The list of nodes to
                retrieve importance scores for.

        Returns:
            dict[int, float]: A dictionary mapping node indexes to their normalized
                importance scores.
        """
        result = dict()
        for index, (_, _, importance, _) in enumerate(nodes):
            result[index] = importance

        # normalize
        min_score = 1
        max_score = 10
        for node_index in result.keys():
            result[node_index] = (
                result[node_index] - min_score
            ) / (max_score - min_score)

        return result

    def _retrieve_dict(
        self, focal_points: list[str], top_k: int
    ) -> dict[str, list[tuple[datetime, str, float, bool]]]:
        """
        Retrieve nodes from the associative memory based on given focal points.

        Args:
            focal_points (list[str]): List of focal points to retrieve nodes for.
            top_k (int): Number of top nodes to retrieve for each focal point.

        Returns:
            dict[str, list[tuple[datetime, str, float, bool]]]: Dictionary mapping
                each focal point to a list of top-k nodes.

        """
        if len(focal_points) == 0:
            focal_points = ["default"]

        nodes = self.associative_memory.read_memory_md(
            self.persona.current_time
        )

        recency_scores = self._recency_retrieval(nodes)
        importance_scores = self._importance_retrieval(nodes)

        acc_nodes = dict()
        combined_scores = dict()
        
        # ANG: I don't understand what giorgio was trying to accomplish here
        # looks like it was a bug here... But probably not important. Functionally it's the same.

        for node_index in recency_scores.keys():
            combined_scores[node_index] = (
                recency_scores[node_index] * self.weights["recency"]
                + importance_scores[node_index] * self.weights["importance"]
            )

        # Put max score to node with always_include flag
        max_value = max(combined_scores.values()) if combined_scores else 10
        for node_index, (_, _, _, always_include) in enumerate(nodes):
            if always_include:
                combined_scores[node_index] = max_value + 1

        # sort by combined scores
        sorted_nodes = sorted(
            enumerate(nodes),
            key=lambda item: combined_scores[item[0]],
            reverse=True,
        )

        # pick top k
        top_k_nodes = [node for _, node in sorted_nodes[:top_k]]

        for focal_point in focal_points:
            acc_nodes[focal_point] = top_k_nodes
        return acc_nodes

    def retrieve(
        self, focal_points: list[str], top_k: int
    ) -> list[tuple[datetime, str]]:
        res = self._retrieve_dict(focal_points, top_k)
        res = res.values()
        res = [node for nodes in res for node in nodes]

        # make sure we don't return the same node twice
        res = set(res)
        res = list(res)
        res_sort = [(created, description) for created, description, _, _ in res]
        # sort by time, most recent last
        res_sort = sorted(res_sort, key=lambda x: x[0])
        return res_sort
