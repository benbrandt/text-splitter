use std::fs;

use fake::{Fake, Faker};
use itertools::Itertools;
use more_asserts::assert_le;
#[cfg(feature = "code")]
use text_splitter::{ChunkConfig, CodeSplitter};

#[cfg(feature = "code")]
#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/code/hashbrown_set_rs.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake();
        let splitter = CodeSplitter::new(
            tree_sitter_rust::language(),
            ChunkConfig::new(max_characters).with_trim(false),
        )
        .unwrap();
        let chunks = splitter.chunks(&text).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks {
            assert_le!(chunk.chars().count(), max_characters);
        }
    }
}

#[cfg(feature = "code")]
#[test]
fn random_chunk_indices_increase() {
    let text = fs::read_to_string("tests/inputs/code/hashbrown_set_rs.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake::<usize>();
        let splitter = CodeSplitter::new(
            tree_sitter_rust::language(),
            ChunkConfig::new(max_characters),
        )
        .unwrap();
        let indices = splitter.chunk_indices(&text).map(|(i, _)| i);

        assert!(indices.tuple_windows().all(|(a, b)| a < b));
    }
}

#[cfg(feature = "code")]
#[test]
fn can_handle_invalid_code() {
    let text = "No code here";

    let splitter = CodeSplitter::new(
        tree_sitter_rust::language(),
        ChunkConfig::new(5).with_trim(false),
    )
    .unwrap();
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks.join(""), text);
}

#[cfg(feature = "code")]
#[test]
fn groups_functions() {
    let text = "
fn fn1() {}
fn fn2() {}
fn fn3() {}
fn fn4() {}";

    let splitter = CodeSplitter::new(tree_sitter_rust::language(), ChunkConfig::new(24)).unwrap();
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        ["fn fn1() {}\nfn fn2() {}", "fn fn3() {}\nfn fn4() {}"]
    );
}

#[cfg(feature = "code")]
#[test]
fn groups_functions_with_children() {
    let text = "
fn fn1() {}
fn fn2() {
    let x = 4;
}
fn fn3() {}
fn fn4() {}";

    let splitter = CodeSplitter::new(tree_sitter_rust::language(), ChunkConfig::new(30)).unwrap();
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        [
            "fn fn1() {}",
            "fn fn2() {\n    let x = 4;\n}",
            "fn fn3() {}\nfn fn4() {}"
        ]
    );
}

#[cfg(feature = "code")]
#[test]
fn functions_overlap() {
    let text = "
fn fn1() {}
fn fn2() {}
fn fn3() {}
fn fn4() {}";

    let splitter = CodeSplitter::new(
        tree_sitter_rust::language(),
        ChunkConfig::new(24).with_overlap(12).unwrap(),
    )
    .unwrap();
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        [
            "fn fn1() {}\nfn fn2() {}",
            "fn fn2() {}\nfn fn3() {}",
            "fn fn3() {}\nfn fn4() {}"
        ]
    );
}
