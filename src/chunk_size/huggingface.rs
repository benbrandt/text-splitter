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

        let pad_id = self.get_padding().map(|params| params.pad_id);

        let offsets = encoding
            .get_ids()
            .iter()
            .zip(encoding.get_offsets())
            // Skip padding tokens at beginning and end so they don't count towards the chunk size
            .skip_while(|&(id, _)| pad_id.map_or(false, |pad_id| id == &pad_id))
            .take_while(|&(id, _)| pad_id.map_or(true, |pad_id| id != &pad_id))
            .map(|(_, (start, end))| *start..*end);

        ChunkSize::from_offsets(offsets, capacity)
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
            ChunkSize::from_offsets([1..3, 4..9, 10..11].into_iter(), &capacity)
        );
    }

    #[test]
    fn returns_offsets_handles_prefix() {
        let tokenizer =
            tokenizers::Tokenizer::from_file("./tests/tokenizers/huggingface.json").unwrap();

        let capacity = 10;
        let offsets = tokenizer.chunk_size("An apple a", &capacity);
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([0..2, 3..8, 9..10].into_iter(), &capacity)
        );
    }

    #[test]
    fn handles_padding() {
        let tokenizer = Tokenizer::from_pretrained("thenlper/gte-small", None).unwrap();
        let capacity = 10;
        let offsets = tokenizer.chunk_size("An apple a", &capacity);
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([0..2, 3..8, 9..10].into_iter(), &capacity)
        );
    }
}
