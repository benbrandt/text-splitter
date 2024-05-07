#[cfg(test)]
mod tests {
    use std::ops::Range;

    use tree_sitter::{Node, Parser, Tree, TreeCursor};

    /// New type around a tree-sitter cursor to allow for implementing an iterator.
    struct CursorOffsets<'cursor> {
        cursor: TreeCursor<'cursor>,
    }

    impl<'cursor> CursorOffsets<'cursor> {
        fn new(cursor: TreeCursor<'cursor>) -> Self {
            Self { cursor }
        }
    }

    impl<'cursor> Iterator for CursorOffsets<'cursor> {
        type Item = (usize, Range<usize>);

        fn next(&mut self) -> Option<Self::Item> {
            // There are children (can call this initially because we don't want the root node)
            if self.cursor.goto_first_child()
                // There are sibling elements to grab because we are the deepest level
                || self.cursor.goto_next_sibling()
                // Go up and over to continue along the tree
                || (self.cursor.goto_parent() && self.cursor.goto_next_sibling())
            {
                Some((
                    self.cursor.depth() as usize,
                    self.cursor.node().byte_range(),
                ))
            } else {
                None
            }
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

        let offsets = CursorOffsets::new(tree.walk()).collect::<Vec<_>>();

        assert_eq!(offsets, naive_offsets(&tree));
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
