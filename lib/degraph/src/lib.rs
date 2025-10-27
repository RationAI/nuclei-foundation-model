use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use std::collections::HashSet;

#[pyfunction]
fn build_spatial_graph(points: PyReadonlyArray2<f32>) -> PyResult<Vec<Vec<(usize, f32)>>> {
    let points_arr = points.as_array();
    let num_points = points_arr.shape()[0];
    let mut adj_graph: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_points];

    if num_points < 3 {
        if num_points == 2 {
            let dx = points_arr[[0, 0]] - points_arr[[1, 0]];
            let dy = points_arr[[0, 1]] - points_arr[[1, 1]];
            let dist = f32::hypot(dx, dy);
            adj_graph[0].push((1, dist));
            adj_graph[1].push((0, dist));
        }
        return Ok(adj_graph);
    }

    let point_vec: Vec<Point2<f32>> = points_arr
        .outer_iter()
        .map(|point| Point2::new(point[0], point[1]))
        .collect();

    let triangulation = DelaunayTriangulation::<_>::bulk_load_stable(point_vec)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    if triangulation.num_vertices() != num_points {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "duplicate points are not supported",
        ));
    }

    let mut seen_edges: HashSet<(usize, usize)> = HashSet::with_capacity(num_points * 3);

    for face in triangulation.inner_faces() {
        let handles = face.vertices();
        for i in 0..3 {
            let p1_handle = handles[i];
            let p2_handle = handles[(i + 1) % 3];

            // spade gives vertex handles with an index that corresponds to the input order
            let p1_idx = p1_handle.index();
            let p2_idx = p2_handle.index();

            let (a, b) = if p1_idx < p2_idx {
                (p1_idx, p2_idx)
            } else {
                (p2_idx, p1_idx)
            };

            if seen_edges.insert((a, b)) {
                let dx = points_arr[[a, 0]] - points_arr[[b, 0]];
                let dy = points_arr[[a, 1]] - points_arr[[b, 1]];
                let dist = f32::hypot(dx, dy);

                adj_graph[a].push((b, dist));
                adj_graph[b].push((a, dist));
            }
        }
    }

    Ok(adj_graph)
}

#[pymodule]
fn degraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_spatial_graph, m)?)?;
    Ok(())
}
