from __future__ import annotations

import jax
import jax.numpy as jnp

from ahead_of_time_compilation_test.pipeline_data import PipelineData
from utils import jax_utils


class AOTGenerator:
    """Erzeugt äquidistante Tschebyscheff-Knoten und skaliert sie auf ein Zielintervall."""

    __data__: PipelineData
    _compiled_generate_nodes: jax.lib.xla_extension.CompiledFunction  # Typ-Hinweis

    # --------------------------------------------------------------------- #
    # Konstruktor – hier erfolgt die einmalige AOT-Kompilation             #
    # --------------------------------------------------------------------- #
    def __init__(self, data: PipelineData) -> None:
        self.__data__ = data

        # ──────────────────────────────────────────────────────────────
        # 1. Fixe, **statische** Parameter für den Kernel
        #    (ändern sich nach der Initialisierung nicht mehr)
        # ──────────────────────────────────────────────────────────────
        node_count: int = self.__data__.node_count
        dtype = self.__data__.data_type

        # ──────────────────────────────────────────────────────────────
        # 2.  Reiner JAX-Kernel, der nur noch das Zielintervall
        #     als **dynamisches** Argument erhält
        # ──────────────────────────────────────────────────────────────
        def _generate_nodes_impl(interval: jnp.ndarray) -> jnp.ndarray:
            """
            interval: jnp.ndarray shape (2,) – Zielintervall [a, b]
            """
            # Schritt 1: Tschebyscheff-Knoten (auf [-1, 1])
            nodes = jnp.arange(1, 2 * node_count + 1, 2, dtype=dtype)
            nodes = nodes * (jnp.pi / (2 * node_count))
            nodes = jnp.cos(nodes)

            # Schritt 2: Auf Zielintervall skalieren
            nodes = jax_utils.rescale_array_to_interval(
                nodes,
                jnp.array([-1, 1], dtype=dtype),
                interval,
            )
            return nodes

        # ──────────────────────────────────────────────────────────────
        # 3.  **Ahead-of-time** kompilieren
        #     Wir übergeben ein Dummy-Intervall, damit XLA die
        #     endgültige ausführbare Funktion bauen kann.
        # ──────────────────────────────────────────────────────────────
        dummy_interval = jnp.array([-1.0, 1.0], dtype=dtype)
        self._compiled_generate_nodes = (
            jax.jit(_generate_nodes_impl)     # → XLA-zulässige HLO
                .lower(dummy_interval)        # → Low-Level-IR
                .compile()                    # → ausführbare Binary
        )

    # --------------------------------------------------------------------- #
    # Öffentliche API – nutzt ausschließlich den vorkompilierten Kernel     #
    # --------------------------------------------------------------------- #
    def generate_nodes(self) -> None:
        """
        Führt den vorbereiteten XLA-Executable aus, legt das Ergebnis
        in `self.__data__.node_array` ab und gibt es zurück.
        """
        # Intervall ggf. von Tuple → jnp.ndarray konvertieren
        interval = self.__data__.interpolation_interval

        nodes = self._compiled_generate_nodes(interval)
        self.__data__.nodes = nodes
