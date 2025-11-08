from functools import partial

import jax
import jax.numpy as jnp

from analysis_of_pipeline_data_on_performance.pipeline_data import PipelineData
from utils import jax_utils


class Generator3:
    __data__: PipelineData

    def __init__(self, data: PipelineData) -> None:
        self.__data__ = data

        # 1️⃣  Reine Rechenfunktion (keine self-Captures!)
        def _generate_nodes_impl_(
            node_count: int, interval: tuple[float, float], data_type
        ) -> jnp.ndarray:
            # Rescale int array for better stability
            nodes: jnp.ndarray = jnp.arange(
                1, 2 * node_count + 1, step=2, dtype=data_type
            )
            nodes = jnp.multiply(nodes, jnp.pi / (2 * node_count))

            nodes = jnp.cos(nodes)

            jax_utils.rescale_array_to_interval(nodes, (-1, 1), interval)

            return nodes

        # 2️⃣  JIT mit statischen Parametern
        _jit = jax.jit(
            _generate_nodes_impl_, static_argnames=("node_count", "data_type")
        )

        # 3️⃣  Beispiel-(Attrappen)-Argumente vorbereiten
        abstract_interval = jax.ShapeDtypeStruct((2,), dtype=data.data_type)

        # 4️⃣  Vorab kompilieren  ➜  ausführbares Callable speichern
        self._compiled_generate_nodes = _jit.lower(
            node_count=data.node_count,
            interval=abstract_interval,
            dtype=data.data_type,
            static_argnames=("node_count", "data_type"),
        ).compile()

    def generate_nodes(self):
        interval = jnp.asarray(self.__data__.interpolation_interval, dtype=jnp.float32)
        return self._compiled_generate_nodes(interval=interval)

    def __repr__(self) -> str:
        return "Node generator for type 1 chebyshev points"
