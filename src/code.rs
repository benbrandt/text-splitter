#[cfg(test)]
mod tests {
    use std::ops::Range;

    use tree_sitter::{Node, Parser, Tree, TreeCursor};

    fn cursor_based_offsets(
        collection: &mut Vec<(usize, Range<usize>)>,
        cursor: &mut TreeCursor<'_>,
    ) {
        let cursor_depth = cursor.depth() as usize;
        if cursor_depth > 0 {
            collection.push((cursor_depth, cursor.node().byte_range()));
        }

        // There are children
        if cursor.goto_first_child()
            // There are sibling elements to grab because we are the deepest level
            || cursor.goto_next_sibling()
            // Go up and over to continue along the tree
            || (cursor.goto_parent() && cursor.goto_next_sibling())
        {
            cursor_based_offsets(collection, cursor);
        }
    }

    /// Checks that the optimized version of the code produces the same results as the naive version.
    #[test]
    fn optimized_code_offsets() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::language())
            .expect("Error loading Rust grammar");
        let source_code = "fn test() {
    let x = 1;
}";
        let tree = parser
            .parse(source_code, None)
            .expect("Error parsing source code");

        let mut cursor = tree.walk();
        let mut a_offsets = vec![];
        cursor_based_offsets(&mut a_offsets, &mut cursor);

        let b_offsets = naive_offsets(&tree);

        assert_eq!(a_offsets, b_offsets);
    }

    fn naive_offsets(tree: &Tree) -> Vec<(usize, Range<usize>)> {
        let root_node = tree.root_node();
        let mut offsets = vec![];
        recursive_naive_offsets(&mut offsets, root_node, 0);
        offsets
    }

    // Basic version to compare an optimized version against. According to the tree-sitter
    // documentation, this is not efficient for large trees. But because it is the easiest
    // to reason about it is a good check for correctness.
    fn recursive_naive_offsets(
        collection: &mut Vec<(usize, Range<usize>)>,
        node: Node<'_>,
        depth: usize,
    ) {
        // We can skip the root node
        if depth > 0 {
            collection.push((depth, node.byte_range()));
        }

        for child in node.children(&mut node.walk()) {
            recursive_naive_offsets(collection, child, depth + 1);
        }
    }
}
