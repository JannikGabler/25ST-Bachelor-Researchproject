# def _compile_jax_callable_(self) -> callable:
#     lambda_vectorized: callable = self._create_jax_lambda_()
#     node_count: int = self._pipeline_data_.node_count
#     data_type: type = self._pipeline_data_.data_type
#
#     dummy_argument = jnp.empty(node_count, dtype=data_type)
#
#     # Ahead-of-time compilation
#     return jax.jit(lambda_vectorized).lower(dummy_argument).compile()
