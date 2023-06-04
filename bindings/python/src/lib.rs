use ::text_splitter::{Characters, ChunkCapacity, TextSplitter};
use pyo3::prelude::*;

/// Custom chunk capacity for python to make it easier to work
/// with python arguments
#[derive(Debug)]
struct PyChunkCapacity {
    start: Option<usize>,
    end: usize,
}

impl PyChunkCapacity {
    fn new(start: Option<usize>, end: usize) -> Self {
        Self { start, end }
    }
}

impl ChunkCapacity for PyChunkCapacity {
    fn start(&self) -> Option<usize> {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }
}

#[pyclass]
struct CharacterTextSplitter {
    splitter: TextSplitter<Characters>,
}

#[pymethods]
impl CharacterTextSplitter {
    #[new]
    #[pyo3(signature = (trim_chunks=false))]
    fn new(trim_chunks: bool) -> Self {
        Self {
            splitter: TextSplitter::default().with_trim_chunks(trim_chunks),
        }
    }

    fn chunks<'text, 'splitter: 'text>(
        &'splitter self,
        text: &'text str,
        chunk_capacity_end: usize,
        chunk_capacity_start: Option<usize>,
    ) -> Vec<&'text str> {
        self.splitter
            .chunks(
                text,
                PyChunkCapacity::new(chunk_capacity_start, chunk_capacity_end),
            )
            .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn text_splitter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CharacterTextSplitter>()?;
    Ok(())
}
