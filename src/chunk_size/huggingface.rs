use tokenizers::Tokenizer;

use crate::ChunkSizer;

impl ChunkSizer for &Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn size(&self, chunk: &str) -> usize {
        let encoding = self
            .encode(chunk, false)
            .expect("Unable to tokenize the following string {chunk}");

        let pad_id = self.get_padding().map(|params| params.pad_id);

        encoding
            .get_ids()
            .iter()
            // Skip padding tokens at beginning and end so they don't count towards the chunk size
            .skip_while(|&id| pad_id.map_or(false, |pad_id| id == &pad_id))
            .take_while(|&id| pad_id.map_or(true, |pad_id| id != &pad_id))
            .count()
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
        assert_eq!(size, 3);
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
        assert_eq!(size, 3);
    }
}
