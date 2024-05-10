use std::fs;

use fake::{Fake, Faker};
use itertools::Itertools;
use more_asserts::assert_le;
#[cfg(feature = "code")]
use text_splitter::{ChunkConfig, ExperimentalCodeSplitter};

#[cfg(feature = "code")]
#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/code/hashbrown_set_rs.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake();
        let splitter = ExperimentalCodeSplitter::new(
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
        let splitter = ExperimentalCodeSplitter::new(
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

    let splitter = ExperimentalCodeSplitter::new(
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

    let splitter =
        ExperimentalCodeSplitter::new(tree_sitter_rust::language(), ChunkConfig::new(24)).unwrap();
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        ["fn fn1() {}\nfn fn2() {}", "fn fn3() {}\nfn fn4() {}"]
    );
}
