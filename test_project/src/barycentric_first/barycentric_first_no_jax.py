import numpy as np

def barycentric_type1_interpolate_no_jax(evaluation_points, interpolation_nodes, function_values):

    interpolation_nodes = np.asarray(interpolation_nodes)
    function_values = np.asarray(function_values)
    evaluation_points = np.atleast_1d(evaluation_points)

    diffs = interpolation_nodes[:, None] - interpolation_nodes[None, :]
    np.fill_diagonal(diffs, 1.0)
    weights = 1.0 / np.prod(diffs, axis=1)

    result = np.empty_like(evaluation_points, dtype=float)

    for i, x in enumerate(evaluation_points):
        diffs_x = x - interpolation_nodes

        if np.any(diffs_x == 0.0):
            result[i] = function_values[np.where(diffs_x == 0.0)[0][0]]
        else:
            terms = weights / diffs_x
            result[i] = np.prod(diffs_x) * np.sum(terms * function_values)

    return result