"""Tests for functions in the optimize submodule."""

from ferrmion.optimize.bonsai import bonsai_algorithm
import rustworkx as rx
import numpy as np
from pytest import fixture


def test_bonsai():
    graph = rx.PyGraph()
    graph.add_nodes_from(range(37))
    graph.add_edges_from_no_data(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (4, 8),
            (5, 9),
            (5, 10),
            (6, 11),
            (6, 12),
            (7, 13),
            (8, 14),
            (9, 15),
            (10, 16),
            (11, 17),
            (12, 18),
            (13, 19),
            (13, 20),
            (14, 21),
            (14, 22),
            (15, 23),
            (15, 24),
            (16, 25),
            (16, 26),
            (17, 27),
            (17, 28),
            (18, 29),
            (18, 30),
            (22, 31),
            (26, 32),
            (30, 33),
            (31, 34),
            (32, 35),
            (33, 36),
        ]
    )

    bonsai_homo = bonsai_algorithm(graph=graph, homogenous=True)
    assert bonsai_homo.as_dict() == {
        "x": {
            "x": {
                "x": {
                    "x": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": {
                            "x": {
                                "x": {"x": None, "y": None, "z": None},
                                "y": None,
                                "z": None,
                            },
                            "y": None,
                            "z": None,
                        },
                        "z": None,
                    },
                    "y": None,
                    "z": None,
                },
                "y": {
                    "x": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": {"x": None, "y": None, "z": None},
                        "z": None,
                    },
                    "y": None,
                    "z": None,
                },
                "z": None,
            },
            "y": None,
            "z": None,
        },
        "y": {
            "x": {
                "x": {
                    "x": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": {"x": None, "y": None, "z": None},
                        "z": None,
                    },
                    "y": None,
                    "z": None,
                },
                "y": {
                    "x": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": {
                            "x": {
                                "x": {"x": None, "y": None, "z": None},
                                "y": None,
                                "z": None,
                            },
                            "y": None,
                            "z": None,
                        },
                        "z": None,
                    },
                    "y": None,
                    "z": None,
                },
                "z": None,
            },
            "y": None,
            "z": None,
        },
        "z": {
            "x": {
                "x": {
                    "x": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": {"x": None, "y": None, "z": None},
                        "z": None,
                    },
                    "y": None,
                    "z": None,
                },
                "y": {
                    "x": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": {
                            "x": {
                                "x": {"x": None, "y": None, "z": None},
                                "y": None,
                                "z": None,
                            },
                            "y": None,
                            "z": None,
                        },
                        "z": None,
                    },
                    "y": None,
                    "z": None,
                },
                "z": None,
            },
            "y": None,
            "z": None,
        },
    }

    bonsai_hetero = bonsai_algorithm(graph=graph, homogenous=False)
    assert bonsai_hetero.as_dict() == {
        "x": {
            "x": None,
            "y": None,
            "z": {
                "x": {
                    "x": None,
                    "y": None,
                    "z": {
                        "x": {
                            "x": None,
                            "y": None,
                            "z": {
                                "x": None,
                                "y": None,
                                "z": {"x": None, "y": None, "z": None},
                            },
                        },
                        "y": None,
                        "z": {"x": None, "y": None, "z": None},
                    },
                },
                "y": None,
                "z": {
                    "x": None,
                    "y": None,
                    "z": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": None,
                        "z": {"x": None, "y": None, "z": None},
                    },
                },
            },
        },
        "y": {
            "x": None,
            "y": None,
            "z": {
                "x": {
                    "x": None,
                    "y": None,
                    "z": {
                        "x": {
                            "x": None,
                            "y": None,
                            "z": {
                                "x": None,
                                "y": None,
                                "z": {"x": None, "y": None, "z": None},
                            },
                        },
                        "y": None,
                        "z": {"x": None, "y": None, "z": None},
                    },
                },
                "y": None,
                "z": {
                    "x": None,
                    "y": None,
                    "z": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": None,
                        "z": {"x": None, "y": None, "z": None},
                    },
                },
            },
        },
        "z": {
            "x": None,
            "y": None,
            "z": {
                "x": {
                    "x": None,
                    "y": None,
                    "z": {
                        "x": {"x": None, "y": None, "z": None},
                        "y": None,
                        "z": {"x": None, "y": None, "z": None},
                    },
                },
                "y": None,
                "z": {
                    "x": None,
                    "y": None,
                    "z": {
                        "x": {
                            "x": None,
                            "y": None,
                            "z": {
                                "x": None,
                                "y": None,
                                "z": {"x": None, "y": None, "z": None},
                            },
                        },
                        "y": None,
                        "z": {"x": None, "y": None, "z": None},
                    },
                },
            },
        },
    }

    assert bonsai_hetero.root_node.child_qubit_labels == {
        "": 0,
        "x": 2,
        "y": 3,
        "z": 1,
        "xz": 5,
        "yz": 6,
        "zz": 4,
        "xzx": 10,
        "xzz": 9,
        "yzx": 12,
        "yzz": 11,
        "zzx": 7,
        "zzz": 8,
        "xzxz": 16,
        "xzzz": 15,
        "yzxz": 18,
        "yzzz": 17,
        "zzxz": 13,
        "zzzz": 14,
        "xzxzx": 26,
        "xzxzz": 25,
        "xzzzx": 23,
        "xzzzz": 24,
        "yzxzx": 30,
        "yzxzz": 29,
        "yzzzx": 28,
        "yzzzz": 27,
        "zzxzx": 20,
        "zzxzz": 19,
        "zzzzx": 22,
        "zzzzz": 21,
        "xzxzxz": 32,
        "yzxzxz": 33,
        "zzzzxz": 31,
        "xzxzxzz": 35,
        "yzxzxzz": 36,
        "zzzzxzz": 34,
    }
