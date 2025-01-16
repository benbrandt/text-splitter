use tokenizers::{Encoding, Tokenizer};

use crate::ChunkSizer;

/// Compute the number of tokens that exist within an entire [`Encoding`] object.
///
/// Take into account [`Encoding::get_overflowing`] for cases where the [`Tokenizer`] producing the [`Encoding`] has truncation parameters set.
fn num_tokens_with_overflow(encoding: &Encoding, pad_id: Option<u32>) -> usize {
    let base = encoding
        .get_ids()
        .iter()
        // Skip padding tokens at beginning and end so they don't count towards the chunk size
        .skip_while(|&id| pad_id.is_some_and(|pad_id| id == &pad_id))
        .take_while(|&id| pad_id.map_or(true, |pad_id| id != &pad_id))
        .count();

    // If the [`Tokenizer`] has truncation, need to check overflow encodings to determine overall size.
    let overflow: usize = encoding
        .get_overflowing()
        .iter()
        .map(|enc| num_tokens_with_overflow(enc, pad_id))
        .sum();

    base + overflow
}

impl ChunkSizer for &Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn size(&self, chunk: &str) -> usize {
        let encoding = self
            .encode(chunk, true)
            .expect("Unable to tokenize the following string {chunk}");

        let pad_id = self.get_padding().map(|params| params.pad_id);
        num_tokens_with_overflow(&encoding, pad_id)
    }
}

impl ChunkSizer for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn size(&self, chunk: &str) -> usize {
        (&self).size(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_size() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let size = tokenizer.size(" An apple a");
        // Bert has a beginning and end token
        assert_eq!(size, 5);
    }

    #[test]
    fn returns_size_handles_prefix() {
        let tokenizer =
            tokenizers::Tokenizer::from_file("./tests/tokenizers/huggingface.json").unwrap();

        let size = tokenizer.size("An apple a");
        assert_eq!(size, 3);
    }

    #[test]
    fn handles_padding() {
        let tokenizer = Tokenizer::from_pretrained("thenlper/gte-small", None).unwrap();
        let size = tokenizer.size("An apple a");
        // Has a beginning and end token
        assert_eq!(size, 5);
    }

    #[test]
    fn handle_truncation() {
        let tokenizer = Tokenizer::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", None)
            .expect("Could not load tokenizer 'sentence-transformers/all-MiniLM-L6-v2'");

        // Need to ensure chunk is large enough to cause Encoding overflows.
        assert_eq!(
            tokenizer.size(" An apple a day keeps the doctor away".repeat(16).as_str()),
            // Overflows at 128, with special tokens at beginning and end of each section of tokens
            132
        );
    }
}
