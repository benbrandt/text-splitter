use tokenizers::{Encoding, Tokenizer};

use crate::ChunkSizer;

/// Compute the number of tokens that exist within an entire [`Encoding`] object.
///
/// Take into account [`Encoding::get_overflowing`] for cases where the [`Tokenizer`] producing the
/// [`Encoding`] has truncation parameters set.
fn num_tokens_with_overflow(encoding: &Encoding) -> usize {
    let base = encoding
        .get_attention_mask()
        .iter()
        // Padding has a zero mask; real content may share its token ID.
        .filter(|&&attention| attention != 0)
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

    fn local_tokenizer(direction: tokenizers::PaddingDirection) -> Tokenizer {
        use tokenizers::{
            models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
            PaddingParams, PaddingStrategy,
        };
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[PAD]".into(), 0),
                    ("[UNK]".into(), 1),
                    ("first".into(), 2),
                    ("second".into(), 3),
                    ("third".into(), 4),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            direction,
            pad_id: 0,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));
        tokenizer
    }

    #[test]
    fn content_padding_ids_are_counted() {
        for direction in [
            tokenizers::PaddingDirection::Left,
            tokenizers::PaddingDirection::Right,
        ] {
            let mut tokenizer = local_tokenizer(direction);
            for padding in [true, false] {
                if !padding {
                    tokenizer.with_padding(None);
                }
                for (text, expected) in [
                    ("", 0),
                    ("first second third", 3),
                    ("[PAD]", 1),
                    ("[PAD] first second third", 4),
                    ("first [PAD] second third", 4),
                    ("first second third [PAD]", 4),
                ] {
                    assert_eq!(
                        tokenizer.size(text),
                        expected,
                        "{direction:?}, padding={padding}, {text:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn content_padding_ids_respect_chunk_budget() {
        use crate::{ChunkConfig, TextSplitter};
        for direction in [
            tokenizers::PaddingDirection::Left,
            tokenizers::PaddingDirection::Right,
        ] {
            let tokenizer = local_tokenizer(direction);
            let splitter = TextSplitter::new(ChunkConfig::new(2).with_sizer(tokenizer.clone()));
            let chunks: Vec<_> = splitter.chunks("first [PAD] second third").collect();
            assert!(chunks.len() > 1);
            for chunk in chunks {
                let encoding = tokenizer.encode_fast(chunk, false).unwrap();
                let content: u32 = encoding.get_attention_mask().iter().sum();
                assert!(content <= 2, "{chunk:?} has {content} real tokens");
            }
        }
    }

    #[test]
    fn content_padding_ids_in_overflow_are_counted() {
        let mut encoding =
            Encoding::from_tokens(vec![tokenizers::Token::new(0, "[PAD]".into(), (0, 5))], 0);
        let mut overflow = Encoding::from_tokens(
            vec![
                tokenizers::Token::new(0, "[PAD]".into(), (0, 5)),
                tokenizers::Token::new(2, "first".into(), (6, 11)),
            ],
            0,
        );
        overflow.pad(8, 0, 0, "[PAD]", tokenizers::PaddingDirection::Left);
        encoding.set_overflowing(vec![overflow]);
        encoding.pad(8, 0, 0, "[PAD]", tokenizers::PaddingDirection::Right);
        assert_eq!(num_tokens_with_overflow(&encoding), 3);
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
