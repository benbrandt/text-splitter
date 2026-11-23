//! Balancing can move a cut, but does not change the number or coverage of chunks.
use std::cell::Cell;

use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter, TryChunkSizer};

const SENTENCES: &str = "Sentence 1. Sentence 2. Sentence 3. Sentence 4.";

#[test]
fn balances_whole_sentences_only_when_enabled() {
    let splitter = TextSplitter::new(36);
    let greedy = ["Sentence 1. Sentence 2. Sentence 3.", "Sentence 4."];
    assert_eq!(splitter.chunks(SENTENCES).collect::<Vec<_>>(), greedy);
    let splitter = splitter.with_balanced(true);
    assert_eq!(
        splitter.chunks(SENTENCES).collect::<Vec<_>>(),
        ["Sentence 1. Sentence 2.", "Sentence 3. Sentence 4."]
    );
    assert_eq!(
        splitter
            .with_balanced(false)
            .chunks(SENTENCES)
            .collect::<Vec<_>>(),
        greedy
    );
}

#[test]
fn preserves_real_overlap_but_can_repair_a_boundary_without_one() {
    let balanced = |overlap| {
        TextSplitter::new(ChunkConfig::new(36).with_overlap(overlap).unwrap())
            .with_balanced(true)
            .chunks(SENTENCES)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        balanced(2),
        ["Sentence 1. Sentence 2.", "Sentence 3. Sentence 4."]
    );
    assert_eq!(
        balanced(15),
        [
            "Sentence 1. Sentence 2. Sentence 3.",
            "Sentence 3. Sentence 4."
        ]
    );
}

#[test]
fn a_range_uses_the_desired_size_not_the_maximum_to_identify_short_chunks() {
    assert_eq!(
        TextSplitter::new(9..=10)
            .with_balanced(true)
            .chunks("12345678901")
            .collect::<Vec<_>>(),
        ["12345", "678901"]
    );
    let text = "123456789012345678";
    let chunks = |balanced| {
        TextSplitter::new(4..=10)
            .with_balanced(balanced)
            .chunks(text)
            .collect::<Vec<_>>()
    };
    assert_eq!(chunks(true), chunks(false));
}

#[test]
fn does_not_break_a_whole_word_to_heal_a_period() {
    let splitter = TextSplitter::new(8).with_balanced(true);
    assert_eq!(
        splitter.chunks("one last.").collect::<Vec<_>>(),
        ["one", "last."]
    );
    assert_eq!(
        splitter.chunks("abcdefgh.").collect::<Vec<_>>(),
        ["abcdefgh", "."]
    );
    assert_eq!(
        splitter.chunks("one last\n.").collect::<Vec<_>>(),
        ["one last", "."]
    );
}

#[cfg(feature = "markdown")]
#[test]
fn protects_intact_markdown() {
    use text_splitter::MarkdownSplitter;
    let item = "•\u{a0}`alpha beta gamma delta epsilon zeta eta theta 0`";
    assert_eq!(
        MarkdownSplitter::new(50)
            .with_balanced(true)
            .chunks(item)
            .collect::<Vec<_>>(),
        ["•", "`alpha beta gamma delta epsilon zeta eta theta 0`"]
    );
    assert_eq!(
        MarkdownSplitter::new(12)
            .with_balanced(true)
            .chunks("Some text\n\n---\n\nwith a rule")
            .collect::<Vec<_>>(),
        ["Some text", "---", "with a rule"]
    );
}

#[cfg(feature = "markdown")]
#[test]
fn character_indices_follow_visible_chunks_even_if_overlap_exposes_indentation() {
    use text_splitter::MarkdownSplitter;
    let text = "\u{a0} \r\n\u{2029}\u{a0}...).ไทย\n-";
    for balanced in [false, true] {
        let splitter = MarkdownSplitter::new(ChunkConfig::new(15).with_overlap(12).unwrap())
            .with_balanced(balanced);
        let indices: Vec<_> = splitter
            .chunk_char_indices(text)
            .map(|c| (c.char_offset, c.chunk))
            .collect();
        assert_eq!(
            indices,
            [(6, "...).ไทย"), (1, " \r\n\u{2029}\u{a0}...).ไทย\n-")]
        );
    }
}

#[cfg(feature = "tiktoken-rs")]
#[test]
fn bpe_sizes_are_not_additive() {
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(3).with_sizer(&bpe)).with_balanced(true);
    let chunks = splitter.chunks("123 ください))").collect::<Vec<_>>();
    assert_eq!(chunks.concat(), "123 ください))");
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().all(|c| bpe.size(c) <= 3));
}

#[test]
fn sizes_both_replacements_even_when_shorter_text_costs_more() {
    struct Expensive(&'static str);
    impl ChunkSizer for Expensive {
        fn size(&self, chunk: &str) -> usize {
            if chunk == self.0 {
                9
            } else {
                chunk.chars().count()
            }
        }
    }
    for expensive in ["one", "last."] {
        let splitter = TextSplitter::new(ChunkConfig::new(8).with_sizer(Expensive(expensive)))
            .with_balanced(true);
        assert_eq!(
            splitter.chunks("one last.").collect::<Vec<_>>(),
            ["one last", "."]
        );
    }
}

#[test]
fn returns_the_valid_chunk_before_a_lookahead_or_candidate_sizing_error() {
    struct Fail<'a>(&'a Cell<bool>, &'a str);
    impl TryChunkSizer for Fail<'_> {
        type Error = ();
        fn try_size(&self, s: &str) -> Result<usize, ()> {
            assert!(!self.0.get(), "called after an error");
            if s == self.1 {
                self.0.set(true);
                Err(())
            } else {
                Ok(s.chars().count())
            }
        }
    }
    for fail_on in [".", "last."] {
        let failed = Cell::new(false);
        let splitter = TextSplitter::new(ChunkConfig::new(8).with_sizer(Fail(&failed, fail_on)))
            .with_balanced(true);
        let mut chunks = splitter.try_chunks("one last. Next one.");
        assert_eq!(chunks.next(), Some(Ok("one last")));
        assert_eq!(chunks.next(), Some(Err(())));
        assert_eq!(chunks.next(), None);
    }
}

#[test]
fn preserves_coverage_indices_and_actual_overlap() {
    let text = "é🦀🌍 one last. More words with e\u{301}. 1234567890\n\n[...] “Καλημέρα!” 你好！";
    for target in 2..=20 {
        for max in [target, target + 5] {
            for overlap in [0, 1, target / 2] {
                for trim in [true, false] {
                    let config = || {
                        ChunkConfig::new(target..=max)
                            .with_overlap(overlap)
                            .unwrap()
                            .with_trim(trim)
                    };
                    let greedy = TextSplitter::new(config())
                        .chunk_indices(text)
                        .collect::<Vec<_>>();
                    let splitter = TextSplitter::new(config()).with_balanced(true);
                    let balanced = splitter.chunk_indices(text).collect::<Vec<_>>();
                    assert_eq!(greedy.len(), balanced.len());
                    let overlaps = |chunks: &[(usize, &str)]| {
                        chunks
                            .windows(2)
                            .filter_map(|w| {
                                let end = w[0].0 + w[0].1.len();
                                (w[1].0 < end).then_some(w[1].0..end)
                            })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(overlaps(&greedy), overlaps(&balanced));
                    let mut end = 0;
                    for (c, (i, s)) in splitter.chunk_char_indices(text).zip(balanced) {
                        assert_eq!(
                            (c.byte_offset, c.char_offset, c.chunk),
                            (i, text[..i].chars().count(), s)
                        );
                        assert_eq!(s, &text[i..i + s.len()]);
                        assert!(s.chars().count() <= max);
                        if i > end {
                            assert!(trim && text[end..i].trim().is_empty());
                        }
                        assert!(i + s.len() > end);
                        end = i + s.len();
                    }
                    assert_eq!(end, text.len());
                }
            }
        }
    }
}
