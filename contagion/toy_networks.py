from __future__ import annotations

from contagion.spec import SystemSpec


def create_no_exposure_network() -> SystemSpec:
    """
    Network with no edges.

    Expected behavior:
        If A fails initially, only A fails.
    """

    return {
        "system_id": "no_exposure_network",
        "systemic_threshold_fraction": 0.5,
        "nodes": [
            {"id": "A", "capital": 10.0},
            {"id": "B", "capital": 10.0},
            {"id": "C", "capital": 10.0},
        ],
        "edges": [],
    }


def create_toy_chain_network() -> SystemSpec:
    """
    Known deterministic cascade:

        Round 0: A fails.
        Round 1: B and C fail.
        Round 2: D fails.
        Round 3: E fails.

    Edge convention:
        source -> target means source default hurts target.
    """

    return {
        "system_id": "toy_chain_network",
        "systemic_threshold_fraction": 0.5,
        "nodes": [
            {"id": "A", "capital": 999.0},
            {"id": "B", "capital": 5.0},
            {"id": "C", "capital": 4.0},
            {"id": "D", "capital": 8.0},
            {"id": "E", "capital": 4.0},
        ],
        "edges": [
            {"source": "A", "target": "B", "exposure": 6.0, "lgd": 1.0},
            {"source": "A", "target": "C", "exposure": 5.0, "lgd": 1.0},
            {"source": "B", "target": "D", "exposure": 4.0, "lgd": 1.0},
            {"source": "C", "target": "D", "exposure": 5.0, "lgd": 1.0},
            {"source": "D", "target": "E", "exposure": 5.0, "lgd": 1.0},
        ],
    }


def create_star_network() -> SystemSpec:
    """
    Highly connected hub example.

    Expected behavior:
        If HUB fails, all leaves fail.
        If one leaf fails, HUB survives.
    """

    return {
        "system_id": "star_network",
        "systemic_threshold_fraction": 0.5,
        "nodes": [
            {"id": "HUB", "capital": 20.0},
            {"id": "A", "capital": 3.0},
            {"id": "B", "capital": 3.0},
            {"id": "C", "capital": 3.0},
            {"id": "D", "capital": 3.0},
        ],
        "edges": [
            {"source": "HUB", "target": "A", "exposure": 4.0, "lgd": 1.0},
            {"source": "HUB", "target": "B", "exposure": 4.0, "lgd": 1.0},
            {"source": "HUB", "target": "C", "exposure": 4.0, "lgd": 1.0},
            {"source": "HUB", "target": "D", "exposure": 4.0, "lgd": 1.0},
            {"source": "A", "target": "HUB", "exposure": 2.0, "lgd": 1.0},
            {"source": "B", "target": "HUB", "exposure": 2.0, "lgd": 1.0},
            {"source": "C", "target": "HUB", "exposure": 2.0, "lgd": 1.0},
            {"source": "D", "target": "HUB", "exposure": 2.0, "lgd": 1.0},
        ],
    }
