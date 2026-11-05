use tokenizers::{Encoding, Tokenizer};

use crate::ChunkSizer;

/// Compute the number of tokens that exist within an entire [`Encoding`] object.
///
/// Take into account [`Encoding::get_overflowing`] for cases where the [`Tokenizer`] producing the
/// [`Encoding`] has truncation parameters set.
fn num_tokens_with_overflow(encoding: &Encoding) -> usize {
    // The attention mask marks padding, so padding doesn't count towards the chunk size while
    // content tokens still do, even when their id matches the pad id.
    let base = encoding
        .get_attention_mask()
        .iter()
        .filter(|&&mask| mask != 0)
        .count();

    // If the [`Tokenizer`] has truncation, need to check overflow encodings to determine overall size.
    let overflow: usize = encoding
        .get_overflowing()
        .iter()
        .map(num_tokens_with_overflow)
        .sum();

    base + overflow
}

impl ChunkSizer for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// If the tokenizer has truncation enabled, the returned size may be truncated by the
    /// tokenizer. Disable truncation on the tokenizer before constructing a splitter if you need
    /// chunk sizes to reflect the full input text.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn size(&self, chunk: &str) -> usize {
        let encoding = self
            .encode_fast(chunk, false)
            .expect("Unable to tokenize the following string {chunk}");

        num_tokens_with_overflow(&encoding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokenizers::{
        models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::WhitespaceSplit,
        utils::{
            padding::{PaddingDirection, PaddingParams, PaddingStrategy},
            truncation::TruncationParams,
        },
    };

    /// Build a deterministic word-level tokenizer that pads to a fixed length and whose
    /// vocabulary has a literal `[PAD]` entry, so text can tokenize to the pad id.
    fn padding_tokenizer(direction: PaddingDirection) -> Tokenizer {
        let mut vocab = ahash::AHashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("first".to_string(), 2);
        vocab.insert("second".to_string(), 3);
        vocab.insert("third".to_string(), 4);

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            direction,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));
        tokenizer
    }

    fn assert_only_content_is_counted(tokenizer: &Tokenizer) {
        // Tokens whose id happens to be the pad id are still content.
        assert_eq!(tokenizer.size("first second third"), 3);
        assert_eq!(tokenizer.size("first [PAD] second third"), 4);
        assert_eq!(tokenizer.size("[PAD] first second third"), 4);
        assert_eq!(tokenizer.size("first second third [PAD]"), 4);
        assert_eq!(tokenizer.size("[PAD]"), 1);
        assert_eq!(tokenizer.size(""), 0);
        assert_eq!(tokenizer.size("   "), 0);
    }

    #[test]
    fn counts_pad_id_content_with_right_padding() {
        assert_only_content_is_counted(&padding_tokenizer(PaddingDirection::Right));
    }

    #[test]
    fn counts_pad_id_content_with_left_padding() {
        assert_only_content_is_counted(&padding_tokenizer(PaddingDirection::Left));
    }

    #[test]
    fn counts_pad_id_content_without_padding() {
        let mut tokenizer = padding_tokenizer(PaddingDirection::Right);
        tokenizer.with_padding(None);
        assert_only_content_is_counted(&tokenizer);
    }

    #[test]
    fn counts_pad_id_content_with_truncation() {
        let mut tokenizer = padding_tokenizer(PaddingDirection::Right);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 2,
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(tokenizer.size("first second third"), 2);
    }

    #[test]
    fn splits_pad_id_content_within_capacity() {
        let text = "first [PAD] second third";
        for direction in [PaddingDirection::Right, PaddingDirection::Left] {
            // Measure the chunks without padding so the expected token counts don't depend on
            // the sizer under test.
            let mut sizer = padding_tokenizer(direction);
            sizer.with_padding(None);
            let splitter = crate::TextSplitter::new(
                crate::ChunkConfig::new(2)
                    .with_sizer(padding_tokenizer(direction))
                    .with_trim(false),
            );

            let chunks = splitter.chunks(text).collect::<Vec<_>>();

            assert_eq!(chunks.len(), 2);
            assert!(chunks.iter().all(|chunk| sizer.size(chunk) <= 2));
            assert_eq!(chunks.concat(), text);
        }
    }

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

    #[test]
    fn returns_truncated_size_when_tokenizer_has_truncation() {
        let tokenizer = Tokenizer::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", None)
            .expect("Could not load tokenizer 'sentence-transformers/all-MiniLM-L6-v2'");

        assert_eq!(
            tokenizer.size("An apple a day keeps the doctor away.".repeat(100).as_str()),
            128
        );
    }

    #[test]
    fn returns_full_size_when_truncation_disabled() {
        let mut tokenizer =
            Tokenizer::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", None)
                .expect("Could not load tokenizer 'sentence-transformers/all-MiniLM-L6-v2'");
        tokenizer
            .with_truncation(None)
            .expect("Could not disable tokenizer truncation");

        assert_eq!(
            tokenizer.size("An apple a day keeps the doctor away.".repeat(100).as_str()),
            900
        );
    }
}
