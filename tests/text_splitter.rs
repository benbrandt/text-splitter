//! Test for `TextSplitter` behavior.
use std::{
    cell::Cell,
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};

use fake::{Fake, Faker};
use itertools::Itertools;
use more_asserts::assert_le;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter};

#[derive(Default)]
struct RecordingWordSizer {
    max_bytes: AtomicUsize,
}

impl RecordingWordSizer {
    fn max_bytes(&self) -> usize {
        self.max_bytes.load(Ordering::SeqCst)
    }
}

impl ChunkSizer for RecordingWordSizer {
    fn size(&self, chunk: &str) -> usize {
        self.max_bytes.fetch_max(chunk.len(), Ordering::SeqCst);
        chunk.split_whitespace().count()
    }
}

fn text_with_distant_paragraph_separator(line_count: usize) -> String {
    let line = "alpha beta gamma delta epsilon zeta eta theta";
    let mut text = String::new();
    for i in 0..line_count {
        if i > 0 {
            text.push('\n');
        }
        text.push_str(line);
    }
    text.push_str("\n\nTHE END.");
    text
}

#[test]
fn chunk_by_paragraphs() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy.";
    let splitter = TextSplitter::new(ChunkConfig::new(21).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert_eq!(
        vec![
            "Mr. Fox jumped.\n[...]",
            "\r\n\r\n",
            "The dog was too lazy."
        ],
        chunks
    );
}

#[test]
fn handles_ending_on_newline() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\n";
    let splitter = TextSplitter::new(ChunkConfig::new(21).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert_eq!(vec!["Mr. Fox jumped.\n[...]", "\r\n\r\n"], chunks);
}

#[test]
fn regex_handles_empty_string() {
    let text = "";
    let splitter = TextSplitter::new(ChunkConfig::new(21).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert!(chunks.is_empty());
}

#[test]
fn double_newline_fallsback_to_single_and_sentences() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy. It just sat there.";
    let splitter = TextSplitter::new(ChunkConfig::new(18).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert_eq!(
        vec![
            "Mr. Fox jumped.\n",
            "[...]\r\n\r\n",
            "The dog was too ",
            "lazy. ",
            "It just sat there."
        ],
        chunks
    );
}

#[test]
fn trim_paragraphs() {
    let text = "Some text\n\nfrom a\ndocument";
    let splitter = TextSplitter::new(10);

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert_eq!(vec!["Some text", "from a", "document"], chunks);
}

#[test]
fn chunk_capacity_range() {
    let text = "12345\n12345";
    let splitter = TextSplitter::new(ChunkConfig::new(5..10).with_trim(false));
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    // \n goes in the second chunk because capacity is reached after first paragraph.
    assert_eq!(vec!["12345", "\n12345"], chunks);
}

#[test]
fn custom_sizer_can_exceed_utf8_byte_length() {
    struct Utf16Bytes;
    impl ChunkSizer for Utf16Bytes {
        fn size(&self, chunk: &str) -> usize {
            chunk.encode_utf16().count() * 2
        }
    }

    // Exercise both an oversized parsed section and the Unicode fallback.
    let text = "Hello.\nHi all.\n🦀🌍!";
    for overlap in [0, 2] {
        let config = ChunkConfig::new(8)
            .with_sizer(Utf16Bytes)
            .with_trim(false)
            .with_overlap(overlap)
            .unwrap();
        let splitter = TextSplitter::new(config);
        let chunks = splitter.chunks(text).collect::<Vec<_>>();
        assert!(
            chunks.iter().all(|chunk| Utf16Bytes.size(chunk) <= 8),
            "{chunks:?}"
        );
        if overlap == 0 {
            assert_eq!(chunks.concat(), text);
        }
    }
}

#[test]
fn indivisible_oversized_chars_are_not_merged() {
    struct MinimumCharge;
    impl ChunkSizer for MinimumCharge {
        fn size(&self, chunk: &str) -> usize {
            chunk.len().max(5)
        }
    }

    let splitter = TextSplitter::new(ChunkConfig::new(4).with_sizer(MinimumCharge));
    assert_eq!(splitter.chunks("abc").collect::<Vec<_>>(), ["a", "b", "c"]);
}

#[test]
fn a_longer_prefix_can_fit_after_an_oversized_first_char() {
    struct PackedPair;
    impl ChunkSizer for PackedPair {
        fn size(&self, chunk: &str) -> usize {
            if chunk == "ab" {
                1
            } else {
                5 + chunk.len()
            }
        }
    }

    let splitter = TextSplitter::new(ChunkConfig::new(5).with_sizer(PackedPair));
    assert_eq!(splitter.chunks("abcdef").next(), Some("ab"));
}

#[test]
fn reuses_the_size_of_a_probed_chunk() {
    struct ProbeCalls(Cell<usize>);

    impl ChunkSizer for ProbeCalls {
        fn size(&self, chunk: &str) -> usize {
            if chunk == "b\nc" {
                self.0.set(self.0.get() + 1);
            }
            chunk.len()
        }
    }

    let calls = ProbeCalls(Cell::new(0));
    let splitter = TextSplitter::new(ChunkConfig::new(2).with_sizer(&calls).with_trim(false));
    let chunks = splitter.chunks("a\nb\nc").collect::<Vec<_>>();

    assert_eq!(chunks, ["a\n", "b\n", "c"]);
    // The probe and the chunk selection both check the same three bytes at offset 2.
    assert_eq!(calls.0.get(), 1);
}

#[test]
fn distant_higher_newline_does_not_size_the_whole_section() {
    let text = text_with_distant_paragraph_separator(1_000);
    let sizer = RecordingWordSizer::default();
    let splitter = TextSplitter::new(ChunkConfig::new(32).with_sizer(&sizer).with_trim(false));

    let chunks = splitter.chunks(&text).collect::<Vec<_>>();

    assert_eq!(chunks.join(""), text);
    assert!(
        sizer.max_bytes() < text.len() / 4,
        "largest sizing call was {} bytes for {} bytes of input",
        sizer.max_bytes(),
        text.len()
    );
}

#[test]
fn distant_fallback_separator_does_not_size_the_whole_section() {
    let text = (0..2000).map(|i| format!("word{i}")).join(" ") + ".";
    let sizer = RecordingWordSizer::default();
    let splitter = TextSplitter::new(ChunkConfig::new(32).with_sizer(&sizer).with_trim(false));

    let chunks = splitter.chunks(&text).collect::<Vec<_>>();

    assert_eq!(chunks.join(""), text);
    assert!(
        sizer.max_bytes() < text.len() / 4,
        "largest sizing call was {} bytes for {} bytes of input",
        sizer.max_bytes(),
        text.len()
    );
}

#[cfg(any(feature = "tokenizers-onig", feature = "tokenizers-fancy-regex"))]
#[test]
fn huggingface_small_chunk_behavior() {
    let tokenizer =
        tokenizers::Tokenizer::from_file("./tests/tokenizers/huggingface.json").unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(5).with_sizer(tokenizer).with_trim(false));

    let text = "notokenexistsforthisword";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["notokenexistsforth", "isword"]);
}

#[cfg(any(feature = "tokenizers-onig", feature = "tokenizers-fancy-regex"))]
#[test]
fn huggingface_tokenizer_with_padding() {
    let tokenizer = tokenizers::Tokenizer::from_pretrained("thenlper/gte-small", None).unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(5).with_sizer(tokenizer).with_trim(false));
    let text = "\nThis is an example text This is an example text\n";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        [
            "\n",
            "This is an example text ",
            "This is an example text\n"
        ]
    );
}

#[cfg(feature = "tiktoken-rs")]
#[test]
fn tiktoken_word_that_fits_is_not_rejected_by_larger_prefixes() {
    let tokenizer = tiktoken_rs::cl100k_base().unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(1).with_sizer(tokenizer).with_trim(false));
    let text = "individual";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, [text]);
}

#[cfg(feature = "tiktoken-rs")]
#[test]
fn tiktoken_punctuation_that_fits_is_not_rejected_by_larger_prefixes() {
    let tokenizer = tiktoken_rs::cl100k_base().unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(1).with_sizer(tokenizer).with_trim(false));
    let text = "!--";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, [text]);
}

#[test]
fn chunk_overlap_characters() {
    let splitter = TextSplitter::new(ChunkConfig::new(4).with_overlap(2).unwrap());
    let text = "1234567890";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["1234", "3456", "5678", "7890"]);
}

#[test]
fn chunk_overlap_words() {
    let splitter = TextSplitter::new(
        ChunkConfig::new(4)
            .with_overlap(3)
            .unwrap()
            .with_trim(false),
    );
    let text = "An apple a day";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["An ", "appl", "pple", " a ", " day"]);
}

#[test]
fn chunk_overlap_words_trim() {
    let splitter = TextSplitter::new(ChunkConfig::new(4).with_overlap(3).unwrap());
    let text = "An apple a day";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["An", "appl", "pple", "a", "day"]);
}

#[test]
fn chunk_overlap_paragraph() {
    let splitter = TextSplitter::new(ChunkConfig::new(14).with_overlap(7).unwrap());
    let text = "Item 1\nItem 2\nItem 3";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["Item 1\nItem 2", "Item 2\nItem 3"]);
}

#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    (0..10).into_par_iter().for_each(|_| {
        let max_characters = Faker.fake::<usize>();
        let splitter = TextSplitter::new(ChunkConfig::new(max_characters).with_trim(false));
        let chunks = splitter.chunks(&text).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.chars().count() <= max_characters));
    });
}

#[test]
fn random_chunk_indices_increase() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    (0..10).into_par_iter().for_each(|_| {
        let max_characters = Faker.fake::<usize>();
        let splitter = TextSplitter::new(ChunkConfig::new(max_characters).with_trim(false));
        let indices = splitter.chunk_indices(&text).map(|(i, _)| i);

        assert!(indices.tuple_windows().all(|(a, b)| a < b));
    });
}

#[test]
fn random_chunk_range() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    (0..10).into_par_iter().for_each(|_| {
        let a = Faker.fake::<Option<u16>>().map(usize::from);
        let b = Faker.fake::<Option<u16>>().map(usize::from);

        let chunks = match (a, b) {
            (None, None) => TextSplitter::new(ChunkConfig::new(..).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
            (None, Some(b)) => TextSplitter::new(ChunkConfig::new(..b).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
            (Some(a), None) => TextSplitter::new(ChunkConfig::new(a..).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
            (Some(a), Some(b)) if b < a => {
                TextSplitter::new(ChunkConfig::new(b..a).with_trim(false))
                    .chunks(&text)
                    .collect::<Vec<_>>()
            }
            (Some(a), Some(b)) => TextSplitter::new(ChunkConfig::new(a..=b).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
        };

        assert_eq!(chunks.join(""), text);
        let max = a.unwrap_or(usize::MIN).max(b.unwrap_or(usize::MAX));
        for chunk in chunks {
            let chars = chunk.chars().count();
            assert_le!(chars, max);
        }
    });
}

#[test]
fn random_chunk_overlap() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    (0..10).into_par_iter().for_each(|_| {
        let a = usize::from(Faker.fake::<u16>());
        let b = usize::from(Faker.fake::<u16>());
        let capacity = a.max(b);
        let overlap = a.min(b);

        let chunks = TextSplitter::new(ChunkConfig::new(capacity).with_overlap(overlap).unwrap())
            .chunks(&text)
            .collect::<Vec<_>>();

        for chunk in chunks {
            let chars = chunk.chars().count();
            assert_le!(chars, capacity);
        }
    });
}

#[test]
fn overlapping_chunks_inside_one_ascii_word() {
    let text = "aB09".repeat(1_024);
    let splitter = TextSplitter::new(ChunkConfig::new(1_000).with_overlap(250).unwrap());
    assert_eq!(
        splitter.chunk_indices(&text).collect::<Vec<_>>(),
        [0, 750, 1500, 2250, 3000, 3750].map(|i| (i, &text[i..(i + 1_000).min(text.len())]))
    );
}
