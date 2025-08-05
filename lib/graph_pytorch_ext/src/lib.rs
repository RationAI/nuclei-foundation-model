use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

#[pyfunction]
fn build_adjacency_graph(
    simplices: PyReadonlyArray2<i64>,
    distances: PyReadonlyArray2<f32>,
    size: usize,
) -> PyResult<Vec<Vec<(usize, f32)>>> {
    let simplices = simplices.as_array();
    let distances = distances.as_array();

    if simplices.shape()[1] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "simplices must be a (N, 3) array",
        ));
    }
    if distances.shape()[1] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "distances must be a (N, 3) array",
        ));
    }
    if simplices.shape()[0] != distances.shape()[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "simplices and distances must have the same number of rows",
        ));
    }

    let num_simplices = simplices.shape()[0];
    let mut adj_graph: Vec<Vec<(usize, f32)>> = vec![Vec::new(); size];

    // Build the graph
    for s in 0..num_simplices {
        for i in 0..3 {
            let p1_idx = simplices[[s, i]] as usize;
            let p2_idx = simplices[[s, (i + 1) % 3]] as usize;
            let dist = distances[[s, i]] as f32;

            if p1_idx < size && p2_idx < size {
                adj_graph[p1_idx].push((p2_idx, dist));
                adj_graph[p2_idx].push((p1_idx, dist));
            }
        }
    }

    Ok(adj_graph)
}

#[pymodule]
fn graph_pytorch_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_adjacency_graph, m)?)?;
    Ok(())
}