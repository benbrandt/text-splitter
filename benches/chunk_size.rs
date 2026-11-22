#![expect(missing_docs)]

use std::{fs, sync::LazyLock};

use ahash::AHashMap;
use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const CHUNK_SIZES: [usize; 3] = [64, 1024, 16384];
const DISTANT_BOUNDARY_CHUNK_SIZES: [usize; 2] = [32, 128];

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const TEXT_FILENAMES: &[&str] = &["romeo_and_juliet", "room_with_a_view"];
const MARKDOWN_FILENAMES: &[&str] = &["commonmark_spec"];
const CODE_FILENAMES: &[&str] = &["hashbrown_set_rs"];

static FILES: LazyLock<AHashMap<&'static str, String>> = LazyLock::new(|| {
    let mut m = AHashMap::new();
    for &name in TEXT_FILENAMES {
        m.insert(
            name,
            fs::read_to_string(format!("tests/inputs/text/{name}.txt")).unwrap(),
        );
    }
    for &name in MARKDOWN_FILENAMES {
        m.insert(
            name,
            fs::read_to_string(format!("tests/inputs/markdown/{name}.md")).unwrap(),
        );
    }
    for &name in CODE_FILENAMES {
        m.insert(
            name,
            fs::read_to_string(format!("tests/inputs/code/{name}.txt")).unwrap(),
        );
    }
    m
});

#[divan::bench_group]
mod text {
    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter};

    use crate::{CHUNK_SIZES, DISTANT_BOUNDARY_CHUNK_SIZES, FILES, TEXT_FILENAMES};

    #[derive(Clone, Copy)]
    struct WordCount;

    impl ChunkSizer for WordCount {
        fn size(&self, chunk: &str) -> usize {
            chunk.split_whitespace().count()
        }
    }

    fn bench<S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> TextSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| (gen_splitter(), FILES.get(filename).unwrap().clone()))
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    fn text_with_distant_paragraph_separator() -> String {
        let line = "alpha beta gamma delta epsilon zeta eta theta";
        let mut text = String::new();
        for i in 0..2_000 {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(line);
        }
        text.push_str("\n\nTHE END.");
        text
    }

    #[divan::bench]
    fn mixed_line_breaks(bencher: Bencher<'_, '_>) {
        let text = "first line\nsecond line\r\nthird line\r\r\n\n".repeat(10_000);
        let splitter = TextSplitter::new(512);
        bencher
            .counter(BytesCount::of_str(&text))
            .bench(|| splitter.chunks(&text).for_each(black_box_drop));
    }

    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || TextSplitter::new(N));
    }

    #[divan::bench(args = [2_000, 4_000, 8_000, 16_000], consts = [0, 24])]
    fn distant_sentence_boundary<const OVERLAP: usize>(bencher: Bencher<'_, '_>, words: usize) {
        let text = "alpha beta gamma delta ".repeat(words / 4);
        let config = ChunkConfig::new(64).with_overlap(OVERLAP).unwrap();
        let splitter = TextSplitter::new(config.with_trim(false));
        bencher
            .counter(BytesCount::of_str(&text))
            .bench(|| splitter.chunks(&text).for_each(black_box_drop));
    }

    #[divan::bench(consts = DISTANT_BOUNDARY_CHUNK_SIZES)]
    fn word_count_distant_paragraph_separator<const N: usize>(bencher: Bencher<'_, '_>) {
        bencher
            .with_inputs(|| {
                (
                    TextSplitter::new(ChunkConfig::new(N).with_sizer(WordCount).with_trim(false)),
                    text_with_distant_paragraph_separator(),
                )
            })
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = ["日本語の長い文章に句読点がない", "ภาษาไทยไม่มีเว้นวรรคระหว่างคำ"])]
    fn dictionary_word_fallback(bencher: Bencher<'_, '_>, phrase: &str) {
        let text = phrase.repeat(500);
        let splitter = TextSplitter::new(64);
        bencher
            .counter(BytesCount::of_str(&text))
            .bench(|| splitter.chunks(&text).for_each(black_box_drop));
    }

    #[divan::bench(args = [100_000, 200_000, 400_000])]
    fn long_ascii_word(bencher: Bencher<'_, '_>, bytes: usize) {
        let text = "a".repeat(bytes);
        let splitter = TextSplitter::new(100);
        bencher
            .counter(BytesCount::of_str(&text))
            .bench(|| splitter.chunks(&text).for_each(black_box_drop));
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        use text_splitter::ChunkConfig;

        bench(bencher, filename, || {
            TextSplitter::new(ChunkConfig::new(N).with_sizer(tiktoken_rs::cl100k_base().unwrap()))
        });
    }

    #[cfg(any(feature = "tokenizers-onig", feature = "tokenizers-fancy-regex"))]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            TextSplitter::new(ChunkConfig::new(N).with_sizer(
                tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            ))
        });
    }
}

#[cfg(feature = "markdown")]
#[divan::bench_group]
mod markdown {
    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkConfig, ChunkSizer, MarkdownSplitter};

    use crate::{CHUNK_SIZES, FILES, MARKDOWN_FILENAMES};

    fn bench<S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> MarkdownSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| (gen_splitter(), FILES.get(filename).unwrap().clone()))
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || MarkdownSplitter::new(N));
    }

    #[divan::bench(args = [true, false])]
    fn single_line_distant_emphasis(bencher: Bencher<'_, '_>, trim: bool) {
        let mut text = "alpha beta gamma delta ".repeat(8_000);
        text.push_str("**end**");
        let splitter = MarkdownSplitter::new(ChunkConfig::new(64).with_trim(trim));
        bencher
            .counter(BytesCount::of_str(&text))
            .bench(|| splitter.chunks(&text).for_each(black_box_drop));
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            MarkdownSplitter::new(
                ChunkConfig::new(N).with_sizer(tiktoken_rs::cl100k_base().unwrap()),
            )
        });
    }

    #[cfg(any(feature = "tokenizers-onig", feature = "tokenizers-fancy-regex"))]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            MarkdownSplitter::new(ChunkConfig::new(N).with_sizer(
                tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            ))
        });
    }
}

#[cfg(feature = "code")]
#[divan::bench_group]
mod code {
    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkConfig, ChunkSizer, CodeSplitter};

    use crate::{CHUNK_SIZES, CODE_FILENAMES, FILES};

    fn bench<S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> CodeSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| (gen_splitter(), FILES.get(filename).unwrap().clone()))
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            CodeSplitter::new(tree_sitter_rust::LANGUAGE, N).unwrap()
        });
    }

    #[divan::bench(args = [1_000, 2_000, 4_000, 8_000])]
    fn repeated_functions(bencher: Bencher<'_, '_>, count: usize) {
        let function = "fn f(x: i32) -> i32 { if x > 0 { x + 1 } else { -x } }\n";
        let source = function.repeat(count);
        let config = ChunkConfig::new(function.len()); // One chunk per function.
        let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, config).unwrap();
        bencher
            .counter(BytesCount::of_str(&source))
            .bench(|| splitter.chunks(&source).for_each(black_box_drop));
    }

    #[divan::bench(args = [2_000, 4_000, 8_000], consts = [6, 160])]
    fn distant_nested_call<const DEPTH: usize>(bencher: Bencher<'_, '_>, count: usize) {
        // The deepest levels occur only at the end, after many chunks of shallow code.
        let source = format!(
            "{}fn g() {{ {}x{}; }}\n",
            "fn f() { a(); }\n".repeat(count),
            "f(".repeat(DEPTH),
            ")".repeat(DEPTH)
        );
        let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, 64).unwrap();
        bencher
            .counter(BytesCount::of_str(&source))
            .bench(|| splitter.chunks(&source).for_each(black_box_drop));
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            CodeSplitter::new(
                tree_sitter_rust::LANGUAGE,
                ChunkConfig::new(N).with_sizer(tiktoken_rs::cl100k_base().unwrap()),
            )
            .unwrap()
        });
    }

    #[cfg(any(feature = "tokenizers-onig", feature = "tokenizers-fancy-regex"))]
    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            CodeSplitter::new(
                tree_sitter_rust::LANGUAGE,
                ChunkConfig::new(N).with_sizer(
                    tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
                ),
            )
            .unwrap()
        });
    }
}
