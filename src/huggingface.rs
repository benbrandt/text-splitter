use std::ops::Range;

use tokenizers::Tokenizer;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        ChunkSize::from_offsets(encoded_offsets(self, chunk), capacity)
    }
}

impl ChunkSizer for &Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        ChunkSize::from_offsets(encoded_offsets(self, chunk), capacity)
    }
}

fn encoded_offsets<'text>(
    tokenizer: &Tokenizer,
    chunk: &'text str,
) -> impl Iterator<Item = Range<usize>> + 'text {
    let encoding = tokenizer
        .encode(chunk, false)
        .expect("Unable to tokenize the following string {chunk}");
    let mut offsets = encoding
        .get_offsets()
        .iter()
        .map(|(start, end)| {
            let end = *end + 1;
            *start..end
        })
        .collect::<Vec<_>>();
    // Sometimes the offsets are off by one because of whitespace prefixing
    let prefixed = offsets
        .last()
        .map(|r| r.end != chunk.len())
        .unwrap_or_default();

    if prefixed {
        for range in &mut offsets {
            if range.start != 0 {
                range.start -= 1;
            }
            range.end -= 1;
        }
    }

    offsets.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let offsets = encoded_offsets(&tokenizer, " An apple a").collect::<Vec<_>>();
        assert_eq!(offsets, vec![0..3, 3..9, 9..11]);
    }

    #[test]
    fn returns_offsets_handles_prefix() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let offsets = encoded_offsets(&tokenizer, "An apple a").collect::<Vec<_>>();
        assert_eq!(offsets, vec![0..2, 2..8, 8..10]);
    }
}
