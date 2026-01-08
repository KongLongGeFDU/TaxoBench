import json
import os


def tree_to_text_tree(node, prefix="    ", is_last=True, is_root=True):
    """
    Convert tree structure to text tree format (excluding leaf node papers, only up to parent of leaf nodes)
    
    Args:
        node: Tree node containing name and subtopics or papers fields, or may be a list
        prefix: Prefix for current line (for indentation), root node defaults to 4 spaces
        is_last: Whether this is the last child node
        is_root: Whether this is the root node
    
    Returns:
        List of text tree strings
    """
    lines = []
    
    # If node is a list, process each element in the list
    if isinstance(node, list):
        if not node:
            return lines
        # If list has only one element, recursively process that element
        if len(node) == 1:
            return tree_to_text_tree(node[0], prefix, is_last, is_root)
        # If list has multiple elements, treat them as child nodes of root
        for i, item in enumerate(node):
            is_last_item = (i == len(node) - 1)
            subtree_lines = tree_to_text_tree(item, prefix, is_last_item, is_root=False)
            lines.extend(subtree_lines)
        return lines
    
    # If node is not a dictionary, return empty
    if not isinstance(node, dict):
        return lines
    
    # Get node name
    name = node.get("name", "")
    
    # Check if node has children
    has_subtopics = "subtopics" in node and node["subtopics"]
    has_papers = "papers" in node and node["papers"]
    
    # Connector for current node
    if is_root:
        # Root node: 4 spaces + name
        lines.append(prefix + name)
        new_prefix = prefix
    else:
        # Non-root node: prefix + connector + name
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + name)
        # Update prefix for child nodes
        new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Process child nodes (subtopics)
    if has_subtopics:
        subtopics = node["subtopics"]
        for i, subtopic in enumerate(subtopics):
            is_last_subtopic = (i == len(subtopics) - 1)
            subtree_lines = tree_to_text_tree(subtopic, new_prefix, is_last_subtopic, is_root=False)
            lines.extend(subtree_lines)
    
    # Note: No longer processing papers (leaf nodes), only keep up to parent of leaf nodes
    # If node only has papers but no subtopics, only output the node itself, not the papers
    
    return lines


def process_jsonl_file(input_file, output_file=None):
    """
    Process jsonl file and convert gt and hierarchy_tree to text tree format
    
    Args:
        input_file: Path to input jsonl file
        output_file: Path to output file (optional, if None prints to console)
    """
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                result = {
                    "id": data.get("id", line_num),
                    "survey_topic": data.get("survey_topic", ""),
                }
                
                # Process gt field
                if "gt" in data:
                    try:
                        gt_tree = tree_to_text_tree(data["gt"])
                        result["gt_text_tree"] = "\n".join(gt_tree)
                    except Exception as e:
                        result["gt_text_tree"] = ""
                
                # Process hierarchy_tree field
                if "hierarchy_tree" in data:
                    try:
                        hierarchy_tree = tree_to_text_tree(data["hierarchy_tree"])
                        result["hierarchy_tree_text_tree"] = "\n".join(hierarchy_tree)
                    except Exception as e:
                        result["hierarchy_tree_text_tree"] = ""
                
                results.append(result)
                
            except json.JSONDecodeError as e:
                continue
    
    # Output results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return results


def process_single_model(model_name: str, model_dir: str):
    """
    Process data for a single model
    
    Args:
        model_name: Model name
        model_dir: Model folder path
    """
    input_file = os.path.join(model_dir, "merged.jsonl")
    output_file = os.path.join(model_dir, "text_trees.jsonl")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        return
    
    try:
        # Process file and save results
        results = process_jsonl_file(input_file, output_file)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def main():
    """
    Batch process data for all models
    """
    # Base path and model list
    base_dir = "model/output"
    models = ["deepseek", "deepseek-thinking", "kimi", "kimi-thinking", "qwen", "qwen-thinking"]
    #models = ["kimi-thinking"]
    
    # Iterate through all models
    for model_name in models:
        model_dir = os.path.join(base_dir, model_name)
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            continue
        
        try:
            process_single_model(model_name, model_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

