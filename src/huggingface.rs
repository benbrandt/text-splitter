use tokenizers::Tokenizer;

use crate::{ChunkSizer, EncodedOffsets};

impl ChunkSizer for Tokenizer {
    /// Return offsets for each unit of text used to calculate chunk size.
    /// Should return an exclusive byte range for each element counted.
    fn encoded_offsets(&self, chunk: &str) -> EncodedOffsets {
        encoded_offsets(self, chunk)
    }

    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, chunk: &str) -> usize {
        chunk_size(self, chunk)
    }
}

impl ChunkSizer for &Tokenizer {
    /// Return offsets for each unit of text used to calculate chunk size.
    /// Should return an exclusive byte range for each element counted.
    fn encoded_offsets(&self, chunk: &str) -> EncodedOffsets {
        encoded_offsets(self, chunk)
    }

    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, chunk: &str) -> usize {
        chunk_size(self, chunk)
    }
}

fn chunk_size(tokenizer: &Tokenizer, chunk: &str) -> usize {
    tokenizer
        .encode(chunk, false)
        .map(|enc| enc.len())
        .expect("Unable to tokenize the following string {str}")
}

fn encoded_offsets(tokenizer: &Tokenizer, chunk: &str) -> EncodedOffsets {
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

    offsets.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let tokenizer = &Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let offsets = tokenizer.encoded_offsets(" An apple a");
        assert_eq!(offsets, vec![0..3, 3..9, 9..11].into());
    }

    #[test]
    fn returns_offsets_handles_prefix() {
        let tokenizer = &Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let offsets = tokenizer.encoded_offsets("An apple a");
        assert_eq!(offsets, vec![0..2, 2..8, 8..10].into());
    }
}
