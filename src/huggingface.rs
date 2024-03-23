use tokenizers::Tokenizer;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for &Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        let encoding = self
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
        let prefixed = offsets.last().is_some_and(|r| r.end != chunk.len());

        if prefixed {
            for range in &mut offsets {
                if range.start != 0 {
                    range.start -= 1;
                }
                range.end -= 1;
            }
        }

        ChunkSize::from_offsets(offsets.into_iter(), capacity)
    }
}

impl ChunkSizer for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let capacity = 10;
        let offsets = tokenizer.chunk_size(" An apple a", &capacity);
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([0..3, 3..9, 9..11].into_iter(), &capacity)
        );
    }

    #[test]
    fn returns_offsets_handles_prefix() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

        let capacity = 10;
        let offsets = tokenizer.chunk_size("An apple a", &capacity);
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([0..2, 2..8, 8..10].into_iter(), &capacity)
        );
    }
}
