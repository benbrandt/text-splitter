#[cfg(test)]
mod tests {
    use std::ops::Range;

    use tree_sitter::{Node, Parser};

    // Basic version to compare an optimized version against
    fn collect_node_and_children(
        collection: &mut Vec<(usize, Range<usize>)>,
        node: Node<'_>,
        depth: usize,
    ) {
        // We can skip the root node
        if depth > 0 {
            collection.push((depth, node.byte_range()));
        }

        for child in node.children(&mut node.walk()) {
            collect_node_and_children(collection, child, depth + 1);
        }
    }

    #[test]
    fn basic_code_offsets() {
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

        let root_node = tree.root_node();
        let mut offsets = vec![];
        collect_node_and_children(&mut offsets, root_node, 0);

        assert_eq!(offsets.len(), 15);
    }
}
