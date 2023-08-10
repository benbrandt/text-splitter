use tiktoken_rs::CoreBPE;

use crate::ChunkSizer;

impl ChunkSizer for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, text: &str) -> usize {
        chunk_size(self, text)
    }
}

impl ChunkSizer for &CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, text: &str) -> usize {
        chunk_size(self, text)
    }
}

fn chunk_size(bpe: &CoreBPE, text: &str) -> usize {
    bpe.encode_ordinary(text).len()
}
