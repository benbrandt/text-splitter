#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

mod chunk_size;
mod splitter;
mod trim;

pub use chunk_size::{
    Characters, ChunkCapacity, ChunkCapacityError, ChunkConfig, ChunkConfigError, ChunkSize,
    ChunkSizer,
};
#[cfg(feature = "markdown")]
pub use splitter::MarkdownSplitter;
pub use splitter::TextSplitter;
