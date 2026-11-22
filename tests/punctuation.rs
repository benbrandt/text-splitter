//! Punctuation repair only changes actual orphans, not word, syntax, or capacity guarantees.
use std::cell::Cell;

use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter, TryChunkSizer};

#[test]
fn closing_marks_stay_with_the_preceding_word() {
    let splitter = TextSplitter::new(8);
    assert_eq!(
        splitter
            .chunk_indices("one last. Next one.")
            .collect::<Vec<_>>(),
        [(0, "one"), (4, "last."), (10, "Next"), (15, "one.")]
    );
    assert_eq!(
        splitter
            .chunk_char_indices("déjà dit。")
            .map(|c| (c.byte_offset, c.char_offset, c.chunk))
            .collect::<Vec<_>>(),
        [(0, 0, "déjà"), (7, 5, "dit。")]
    );
    // Leave the period alone rather than break an intact word.
    assert_eq!(
        splitter.chunks("abcdefgh.").collect::<Vec<_>>(),
        ["abcdefgh", "."]
    );
}

#[test]
fn overlap_can_already_supply_the_word() {
    let splitter = TextSplitter::new(ChunkConfig::new(8).with_overlap(5).unwrap());
    assert_eq!(
        splitter.chunks("one last. Next one.").collect::<Vec<_>>(),
        ["one last", "last.", "Next one", "one."]
    );
}

#[cfg(feature = "markdown")]
#[test]
fn opening_marks_join_forward_without_breaking_intact_markdown() {
    use text_splitter::MarkdownSplitter;

    let splitter = MarkdownSplitter::new(30);
    let intro = "CHAPTER 1\n\n\"Well, Prince, so Genoa and Lucca are now just family estates";
    assert_eq!(
        splitter.chunks(intro).take(2).collect::<Vec<_>>(),
        ["CHAPTER 1", "\"Well, Prince, so Genoa and"]
    );
    let item = "•\u{a0}`alpha beta gamma delta epsilon zeta eta theta 0`";
    assert_eq!(
        splitter.chunks(item).next(),
        Some("•\u{a0}`alpha beta gamma delta")
    );
    // The code span now fits intact, but not with its bullet. Don't break it.
    assert_eq!(
        MarkdownSplitter::new(50).chunks(item).collect::<Vec<_>>(),
        ["•", "`alpha beta gamma delta epsilon zeta eta theta 0`"]
    );
    assert_eq!(
        MarkdownSplitter::new(12)
            .chunks("Some text\n\n---\n\nwith a rule")
            .collect::<Vec<_>>(),
        ["Some text", "---", "with a rule"]
    );
}

#[test]
fn both_replacements_must_fit_the_actual_sizer() {
    struct Expensive<'a>(&'a str);
    impl ChunkSizer for Expensive<'_> {
        fn size(&self, s: &str) -> usize {
            if s == self.0 {
                9
            } else {
                s.chars().count()
            }
        }
    }
    for expensive in ["one", "last."] {
        let splitter = TextSplitter::new(ChunkConfig::new(8).with_sizer(Expensive(expensive)));
        assert_eq!(
            splitter.chunks("one last.").collect::<Vec<_>>(),
            ["one last", "."]
        );
    }
}

#[cfg(feature = "tiktoken-rs")]
#[test]
fn bpe_can_charge_more_for_a_shorter_prefix() {
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    assert_eq!((bpe.size("123 ください"), bpe.size("123 くだ")), (3, 4));
    let splitter = TextSplitter::new(ChunkConfig::new(3).with_sizer(&bpe));
    let chunks = splitter.chunks("123 ください))").collect::<Vec<_>>();
    assert!(chunks.iter().all(|s| bpe.size(s) <= 3));
    assert_eq!(chunks.concat(), "123 ください))");
}

#[test]
fn a_repair_error_preserves_prior_output_and_ends_iteration() {
    struct Fail<'a>(&'a Cell<bool>);
    impl TryChunkSizer for Fail<'_> {
        type Error = ();
        fn try_size(&self, s: &str) -> Result<usize, ()> {
            assert!(!self.0.get(), "called after an error");
            if s == "last." {
                self.0.set(true);
                Err(())
            } else {
                Ok(s.chars().count())
            }
        }
    }
    let failed = Cell::new(false);
    let splitter = TextSplitter::new(ChunkConfig::new(8).with_sizer(Fail(&failed)));
    let mut chunks = splitter.try_chunks("one last. Next one.");
    assert_eq!(chunks.next(), Some(Ok("one last")));
    assert_eq!(chunks.next(), Some(Err(())));
    assert_eq!(chunks.next(), None);
    assert_eq!(chunks.next(), None);
}

#[test]
fn repairs_preserve_source_coverage_and_offsets() {
    let text = "“a café.” Do it.)\r\n\r\n----\n5 €\u{a0}！ ¿Sí?\n👩‍🌾 e\u{301} 你好。   Done.";
    for capacity in 2..=24 {
        for overlap in 0..capacity {
            for trim in [false, true] {
                let config = ChunkConfig::new(capacity)
                    .with_overlap(overlap)
                    .unwrap()
                    .with_trim(trim);
                let splitter = TextSplitter::new(config);
                let mut start = None;
                let mut end = 0;
                for c in splitter.chunk_char_indices(text) {
                    let next_end = c.byte_offset + c.chunk.len();
                    assert_eq!(c.chunk, &text[c.byte_offset..next_end]);
                    assert_eq!(c.char_offset, text[..c.byte_offset].chars().count());
                    assert!(c.chunk.chars().count() <= capacity);
                    // Trimming can expose the same start in overlapping chunks.
                    assert!(start.is_none_or(|previous| c.byte_offset >= previous));
                    assert!(next_end > end);
                    if end < c.byte_offset {
                        assert!(trim && text[end..c.byte_offset].trim().is_empty());
                    } else if overlap == 0 {
                        assert_eq!(end, c.byte_offset);
                    }
                    start = Some(c.byte_offset);
                    end = next_end;
                }
                assert_eq!(end, text.len());
            }
        }
    }
}
