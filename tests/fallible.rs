//! Fallible sizing preserves successful output and terminates on the first error.

use std::{cell::Cell, rc::Rc};

#[cfg(feature = "code")]
use text_splitter::CodeSplitter;
#[cfg(feature = "markdown")]
use text_splitter::MarkdownSplitter;
use text_splitter::{
    Characters, ChunkCapacity, ChunkConfig, ChunkSizer, TextSplitter, TryChunkSizer,
};

#[derive(Debug, PartialEq, Eq)]
struct SizingError(usize);

struct CountingSizer {
    calls: Rc<Cell<usize>>,
    fail_at: Option<usize>,
}

impl TryChunkSizer for CountingSizer {
    type Error = SizingError;

    fn try_size(&self, chunk: &str) -> Result<usize, Self::Error> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            Err(SizingError(call))
        } else {
            Ok(Characters.size(chunk))
        }
    }
}

fn assert_terminal_error<T>(
    mut chunks: impl Iterator<Item = Result<T, SizingError>>,
    calls: &Cell<usize>,
    fail_at: usize,
) {
    assert_eq!(chunks.find_map(Result::err), Some(SizingError(fail_at)));
    assert_eq!(calls.get(), fail_at);
    assert!(chunks.next().is_none());
    assert!(chunks.next().is_none());
    assert_eq!(calls.get(), fail_at);
}

macro_rules! splitter_tests {
    ($module:ident, $splitter:ident, $new:expr_2021) => {
        mod $module {
            use super::*;

            fn make<S: TryChunkSizer>(
                capacity: impl Into<ChunkCapacity>,
                overlap: usize,
                trim: bool,
                sizer: S,
            ) -> $splitter<S> {
                let mut config = ChunkConfig::new(capacity).with_sizer(sizer).with_trim(trim);
                if overlap > 0 {
                    config = config.with_overlap(overlap).unwrap();
                }
                ($new)(config)
            }

            #[test]
            fn successful_output_matches_infallible_methods() {
                for text in [
                    "",
                    " \n\n ",
                    "é🦀a",
                    "e\u{301}\r\n\u{2003}\r\na",
                    "fn café() {\n    let x = \"日本語\";\n}\n\n// next\nfn second() {}\n",
                    "# Title\n\nFirst paragraph. Second sentence.\n\n- One\n- Two\n",
                ] {
                    for capacity in [1, 4, 12, 32] {
                        for overlap in [0, 1, 3] {
                            if overlap > 0 && overlap >= capacity {
                                continue;
                            }
                            for trim in [false, true] {
                                let native = make(capacity, overlap, trim, Characters);
                                let fallible = make(
                                    capacity,
                                    overlap,
                                    trim,
                                    CountingSizer {
                                        calls: Rc::default(),
                                        fail_at: None,
                                    },
                                );
                                assert_eq!(
                                    fallible
                                        .try_chunks(text)
                                        .collect::<Result<Vec<_>, _>>()
                                        .unwrap(),
                                    native.chunks(text).collect::<Vec<_>>(),
                                );
                                assert_eq!(
                                    fallible
                                        .try_chunk_indices(text)
                                        .collect::<Result<Vec<_>, _>>()
                                        .unwrap(),
                                    native.chunk_indices(text).collect::<Vec<_>>(),
                                );
                                assert_eq!(
                                    fallible
                                        .try_chunk_char_indices(text)
                                        .collect::<Result<Vec<_>, _>>()
                                        .unwrap(),
                                    native.chunk_char_indices(text).collect::<Vec<_>>(),
                                );
                            }
                        }
                    }
                }
            }

            #[test]
            fn stops_at_every_sizing_failure() {
                let text = "fn café() {\n    let x = \"日本語\";\n}\n\n// next\nfn second() {}\n";
                for (capacity, overlap) in [
                    (ChunkCapacity::new(8), 0),
                    (ChunkCapacity::new(8), 3),
                    (ChunkCapacity::new(5).with_max(8).unwrap(), 3),
                ] {
                    let calls = Rc::new(Cell::new(0));
                    let baseline = make(
                        capacity,
                        overlap,
                        true,
                        CountingSizer {
                            calls: calls.clone(),
                            fail_at: None,
                        },
                    );
                    baseline
                        .try_chunks(text)
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    let total_calls = calls.get();
                    assert!(total_calls > 1);

                    // Exercise errors at every sizing call, including probing,
                    // chunk selection, subsequent chunks, and overlap.
                    for fail_at in 1..=total_calls {
                        for method in 0..3 {
                            calls.set(0);
                            let splitter = make(
                                capacity,
                                overlap,
                                true,
                                CountingSizer {
                                    calls: calls.clone(),
                                    fail_at: Some(fail_at),
                                },
                            );
                            match method {
                                0 => assert_terminal_error(
                                    splitter.try_chunks(text),
                                    &calls,
                                    fail_at,
                                ),
                                1 => assert_terminal_error(
                                    splitter.try_chunk_indices(text),
                                    &calls,
                                    fail_at,
                                ),
                                _ => assert_terminal_error(
                                    splitter.try_chunk_char_indices(text),
                                    &calls,
                                    fail_at,
                                ),
                            }
                        }
                    }
                }
            }

            #[test]
            fn splitter_can_be_reused_after_error() {
                let calls = Rc::new(Cell::new(0));
                let splitter = make(
                    4,
                    0,
                    false,
                    CountingSizer {
                        calls: calls.clone(),
                        fail_at: Some(1),
                    },
                );
                assert_terminal_error(splitter.try_chunks("abcdef"), &calls, 1);
                let chunks = splitter
                    .try_chunks("abcdef")
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(chunks.concat(), "abcdef");
            }

            #[test]
            fn empty_input_does_not_call_sizer() {
                let calls = Rc::new(Cell::new(0));
                let splitter = make(
                    4,
                    0,
                    true,
                    CountingSizer {
                        calls: calls.clone(),
                        fail_at: Some(1),
                    },
                );
                assert!(splitter.try_chunks("").next().is_none());
                assert!(splitter.try_chunk_indices("").next().is_none());
                assert!(splitter.try_chunk_char_indices("").next().is_none());
                assert_eq!(calls.get(), 0);
            }

            #[test]
            fn zero_capacity_propagates_sizing_error() {
                let calls = Rc::new(Cell::new(0));
                let splitter = make(
                    0,
                    0,
                    true,
                    CountingSizer {
                        calls: calls.clone(),
                        fail_at: Some(1),
                    },
                );
                assert_terminal_error(splitter.try_chunks("abcdef"), &calls, 1);
            }

            #[test]
            fn existing_generic_callers_remain_compatible() {
                fn check<S: ChunkSizer>(sizer: S) {
                    let splitter = make(4, 0, false, sizer);
                    assert_eq!(
                        splitter.chunks("abcdefgh").collect::<Vec<_>>(),
                        ["abcd", "efgh"]
                    );
                    assert_eq!(
                        splitter
                            .try_chunks("abcdefgh")
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap(),
                        splitter.chunks("abcdefgh").collect::<Vec<_>>(),
                    );
                }
                check(Characters);
                check(&Characters);
                check(Box::new(Characters));
                check(Rc::new(Characters));
                check(std::sync::Arc::new(Characters));
            }
        }
    };
}

splitter_tests!(text, TextSplitter, TextSplitter::new);
#[cfg(feature = "markdown")]
splitter_tests!(markdown, MarkdownSplitter, MarkdownSplitter::new);
#[cfg(feature = "code")]
splitter_tests!(code, CodeSplitter, |config| CodeSplitter::new(
    tree_sitter_rust::LANGUAGE,
    config
)
.unwrap());
