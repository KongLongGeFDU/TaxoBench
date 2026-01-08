import os
import json
import time
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://chatapi.littlewheat.com/v1"

# Initialize client
client = OpenAI()


def chat_gpt_call(text, model='gpt-4o'):
    """Call OpenAI's chat completions API"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior, rigorous, and objective research paper evaluation expert. You specialize in constructing and evaluating complex academic taxonomies. Your core capability is utilizing 'Human Expert Knowledge Trees' as the gold standard to conduct strict, detailed, and semantic-based differential evaluations of 'Model-Generated Knowledge Trees,' providing precise scores and clear argumentation."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    # Get response content
    if getattr(completion.choices[0].message, 'content', None):
        content = completion.choices[0].message.content.strip()
        return content
    else:
        return None


def build_judge_prompt(example_tree: str, ground_truth: str, full_score: int = 5) -> str:
    """Build evaluation prompt"""
    return f"""
# Task Description
In an academic Survey generation task, we require an AI model to read a large number of papers and generate a "Taxonomy Tree" that summarizes the knowledge structure of the field.
Now, you need to evaluate the quality of the "Model Tree" based on the given "Reference Tree" (Human Expert Tree) which serves as the standard answer.

# Input Data
Human Expert Tree (Reference):
<reference_tree>
{ground_truth}
</reference_tree>

Model Generated Tree (Model Prediction):
<model_tree>
{example_tree}
</model_tree>

# Evaluation Criteria
Please compare the Model Tree with the Reference Tree based on the following four dimensions and score the Model Tree (1-5 scale).

<criteria_list>
1. Semantic Coverage & Recall
  - Definition: Measures whether the Model Tree contains the core concepts and main branches present in the Reference Tree.
  - Scoring Rubric:
    - 1 (Critical Failure): Misses more than 50% of the core branches (Level 1/Level 2); key concepts are seriously lacking; fails to reflect the overview of the field.
    - 2 (Poor): Covers the main fields but misses a large number of important sub-fields; or exhibits significant conceptual deviation.
    - 3 (Fair): Recalls most Level 1 concepts, but falls short of the Reference Tree in terms of depth or breadth in specific branches; contains moderate omissions.
    - 4 (Good): Recalls the vast majority of core concepts (>90%); even if terminology differs, the semantics correspond to the Reference Tree; only minimal secondary branches are missing.
    - 5 (Excellent): Perfectly covers all conceptual levels of the Reference Tree without omission; or provides extremely valuable and logical supplements on top of full coverage.

2. Sibling Organization (MECE Principle)
  - Definition: Evaluates whether the set of child nodes under the same parent node follows the MECE principle (Mutually Exclusive, Collectively Exhaustive).
  - Scoring Rubric:
    - 1 (Chaotic): Severe semantic overlap between sibling nodes (>50%); or completely lacks classification logic with extremely chaotic vocabulary dimensions.
    - 2 (Poor): Inconsistent classification standards (e.g., mixing methods, applications, datasets, and other dimensions); or the division of a certain category is overly fragmented.
    - 3 (Fair): Overall classification is acceptable, but there are fuzzy boundaries between individual nodes, or mutual exclusivity is not strict enough.
    - 4 (Clear): Clear boundaries between sibling nodes with good mutual exclusivity; classification logic is highly similar to the Reference Tree.
    - 5 (Precise): Node organization is extremely rigorous; classification dimensions are unified and complete, reaching the standard of expert classification with no logical redundancy.

3. Hierarchical Consistency
  - Definition: Evaluates the logical correctness of the "Parent Node -> Child Node" path. The child node must be a proper subset of the parent node (Is-A or Part-Of relationship).
  - Scoring Rubric:
    - 1 (Logical Error): Contains a large number of "inverted" relationships (parent node is more specific than child node); or child nodes do not belong to the parent category at all (severe hallucination).
    - 2 (Hierarchical Confusion): Frequent misalignment of abstract levels; or forced parent-child relationships.
    - 3 (Basic Flow): Most paths are logically valid, but there are minor issues with hierarchical definitions in deeper nodes.
    - 4 (Good Logic): All parent-child relationships are academically valid; consistency of abstract levels is good.
    - 5 (Rigorous Logic): All paths conform to strict academic definitions; the hierarchical progression perfectly matches the logical depth of the Reference Tree, leaving no room for critique.

4. Structural Topology
  - Definition: Evaluates whether the "shape" of the Model Tree is similar to the Reference Tree. Focuses on whether the distribution of depth and breadth is reasonable.
  - Scoring Rubric:
    - 1 (Severe Deformation): Extreme structural difference (e.g., Reference Tree is deep, but Model Tree is a flat list; or vice versa).
    - 2 (Imbalanced): Certain branches are overly expanded (too deep) while others are not expanded (too shallow), causing the overall center of gravity to deviate significantly from the Reference Tree.
    - 3 (Acceptable): The overall shape is roughly similar, but the granularity in certain sub-trees is either too fine or too coarse.
    - 4 (Approximate): The overall depth and the lushness of various branches are consistent with the Reference Tree.
    - 5 (Structural Fit): Perfectly replicates the granularity distribution and cognitive complexity of the Reference Tree; the structure is balanced and aesthetically pleasing.
</criteria_list>

# Instructions
Your task is to strictly compare the `<model_tree>` with the `<reference_tree>` based on each dimension in `<criteria_list>`.
For each dimension, you need to:
1. Evidence Extraction: Identify specific nodes/structures in the Model Tree that support your judgment, citing the corresponding parts of the Reference Tree as a control.
2. Gap Analysis: Clearly point out what the Model Tree got right (Match), and what it got wrong (Mismatch/Hallucination/Omission).
3. Final Scoring: Provide an objective score (1-5) based on your analysis.

# Output Format Requirements
Please strictly follow the `<output_format>` below. Do not include any irrelevant intro or summary. Ensure the output is valid JSON.

<output_format>
{{
  "semantic_coverage": {{
    "score": [Specific Score 1-5],
    "reasoning": "Detailed analysis of Semantic Coverage. E.g., The model tree covers branches [A, B] of the reference tree but seriously misses branch [C]..."
  }},
  "sibling_organization": {{
    "score": [Specific Score 1-5],
    "reasoning": "Detailed analysis of Sibling Organization. E.g., Under the 'Pre-training Models' node, the sibling nodes [X, Y] generated by the model show obvious overlap..."
  }},
  "hierarchical_logic": {{
    "score": [Specific Score 1-5],
    "reasoning": "Detailed analysis of Hierarchical Logic. E.g., The model places the specific 'BERT' at the top level, which contradicts the logical hierarchy of the reference tree..."
  }},
  "structural_topology": {{
    "score": [Specific Score 1-5],
    "reasoning": "Detailed analysis of Structural Topology. E.g., The average depth of the model tree is 2 layers, while the reference tree is 4 layers; the model tree is too flat..."
  }}
}}
</output_format>
Now, please begin the evaluation.
"""


def evaluate_taxonomy_llm(example_tree: str, ground_truth: str, model: str = 'gpt-4o', full_score: int = 5):
    """Evaluate taxonomy using LLM"""
    prompt = build_judge_prompt(example_tree, ground_truth, full_score=full_score)
    for attempt in range(2):
        try:
            content = chat_gpt_call(prompt, model=model)
            if content is None:
                continue
            
            # Error handling: remove possible code block fences and extract JSON body
            text = content.strip()
            # Remove ```json ... ``` or ``` ... ``` wrapping
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            # Extract first JSON object
            match = re.search(r"\{[\s\S]*\}", text)
            json_str = match.group(0) if match else text
            result = json.loads(json_str)
            keys = [
                "semantic_coverage",
                "sibling_organization",
                "hierarchical_logic",
                "structural_topology",
            ]
            cleaned = {}
            for key in keys:
                value = result.get(key, {})
                if isinstance(value, dict):
                    cleaned[key] = float(value.get("score", 0.0))
                else:
                    cleaned[key] = float(value or 0.0)
            return cleaned, content
        except Exception as e:
            if attempt < 1:
                time.sleep(1)
    return None, content if 'content' in locals() else ""


def process_text_trees_file(input_file, output_file, model='gpt-4o', full_score=5, model_response_file=None):
    """
    Process text tree file and evaluate each survey
    
    Args:
        input_file: Path to input jsonl file containing gt_text_tree and hierarchy_tree_text_tree
        output_file: Path to output jsonl file containing original data and scoring results
        model: Model name to use
        full_score: Full score
        model_response_file: Path to save GPT responses (optional, if None uses default path)
    
    Returns:
        results: List of processing results
        avg_scores: Dictionary of average scores for four dimensions, returns None if no valid scores
    """
    results = []
    
    # Set model_response.jsonl file path
    if model_response_file is None:
        model_response_file = os.path.join(os.path.dirname(output_file), "judge_model_response.jsonl")
    os.makedirs(os.path.dirname(model_response_file), exist_ok=True)
    
    # Clear model_response file if it exists
    if os.path.exists(model_response_file):
        open(model_response_file, 'w').close()
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    for line_num, line in enumerate(tqdm(lines, desc="Processing"), 1):
        try:
            data = json.loads(line.strip())
            
            # Get text trees
            example_tree = data.get("hierarchy_tree_text_tree", "")  # Model-generated tree
            ground_truth = data.get("gt_text_tree", "")  # Ground truth tree
            
            # Check for missing data in detail
            missing_fields = []
            if not example_tree:
                missing_fields.append("hierarchy_tree_text_tree")
            if not ground_truth:
                missing_fields.append("gt_text_tree")
            
            if missing_fields:
                # Even if data is missing, record in results and mark as error
                result = {
                    "id": data.get("id", line_num),
                    "survey_topic": data.get("survey_topic", ""),
                    "gt_text_tree": ground_truth if ground_truth else "",
                    "hierarchy_tree_text_tree": example_tree if example_tree else "",
                    "scores": None,
                    "raw_response": "",
                    "error": f"Missing text tree data: {', '.join(missing_fields)}"
                }
                results.append(result)
                # Save to output file
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    for r in results:
                        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            
            # Perform evaluation
            scores, raw_response = evaluate_taxonomy_llm(
                example_tree, 
                ground_truth, 
                model=model, 
                full_score=full_score
            )
            
            # Build result
            result = {
                "id": data.get("id", line_num),
                "survey_topic": data.get("survey_topic", ""),
                "gt_text_tree": ground_truth,
                "hierarchy_tree_text_tree": example_tree,
            }
            
            if scores:
                result["scores"] = scores
                result["raw_response"] = raw_response
            else:
                result["scores"] = None
                result["raw_response"] = raw_response
                result["error"] = "Score parsing failed"
            
            results.append(result)
            
            # Save GPT response to model_response.jsonl
            response_record = {
                "id": data.get("id", line_num),
                "survey_topic": data.get("survey_topic", ""),
                "model_response": raw_response
            }
            with open(model_response_file, 'a', encoding='utf-8') as f_response:
                f_response.write(json.dumps(response_record, ensure_ascii=False) + "\n")
            
            # Save after processing each record (prevent data loss on interruption)
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for r in results:
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            
        except json.JSONDecodeError as e:
            continue
        except Exception as e:
            continue
    
    # Statistics of scoring results
    avg_scores = None
    if results:
        valid_scores = [r["scores"] for r in results if r.get("scores")]
        if valid_scores:
            print("\nScoring statistics:")
            avg_scores = {}
            for key in ["semantic_coverage", "sibling_organization", "hierarchical_logic", "structural_topology"]:
                values = [s[key] for s in valid_scores]
                avg = sum(values) / len(values)
                avg_scores[key] = round(avg, 2)
                print(f"  {key}: Average = {avg:.2f}, Max = {max(values):.2f}, Min = {min(values):.2f}")
    
    return results, avg_scores


def process_single_model(model_name: str, model_dir: str, llm_model='gpt-4o', full_score=5):
    """
    Process data for a single model
    
    Args:
        model_name: Model name
        model_dir: Model folder path
        llm_model: LLM model name to use
        full_score: Full score
    
    Returns:
        dict: Dictionary containing model name and average scores for four dimensions, returns None if processing fails
    """
    input_file = os.path.join(model_dir, "text_trees.jsonl")
    output_file = os.path.join(model_dir, "text_tree_scores.jsonl")
    model_response_file = os.path.join(model_dir, "judge_model_response.jsonl")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        return None
    
    try:
        # Process file and perform evaluation
        results, avg_scores = process_text_trees_file(
            input_file=input_file,
            output_file=output_file,
            model=llm_model,
            full_score=full_score,
            model_response_file=model_response_file
        )
        
        # Return model name and average scores
        if avg_scores:
            return {
                "model_name": model_name,
                **avg_scores
            }
        else:
            return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Batch process data for all models using 6 threads in parallel
    """
    # Base path and model list
    base_dir = "model/output"
    models = ["deepseek", "deepseek-thinking", "kimi", "kimi-thinking", "qwen", "qwen-thinking"]
    
    # Collect results for all models
    all_results = []
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
        future_to_model = {}
        for model_name in models:
            model_dir = os.path.join(base_dir, model_name)
            
            # Check if model directory exists
            if not os.path.exists(model_dir):
                continue
            
            # Submit task to thread pool
            future = executor.submit(process_single_model, model_name, model_dir, 'gpt-4o', 5)
            future_to_model[future] = model_name
        
        # Collect results
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
    
    # Save results to specified file
    output_file = "model/output/results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Print summary information
    if all_results:
        print("\nAverage scores for four dimensions by model:")
        print("-" * 80)
        print(f"{'Model Name':<20} {'Semantic Coverage':<18} {'Sibling Org':<12} {'Hierarchical Logic':<18} {'Structural Topology':<18}")
        print("-" * 80)
        for result in all_results:
            model_name = result.get("model_name", "N/A")
            semantic = result.get("semantic_coverage", 0)
            sibling = result.get("sibling_organization", 0)
            hierarchical = result.get("hierarchical_logic", 0)
            structural = result.get("structural_topology", 0)
            print(f"{model_name:<20} {semantic:<18.2f} {sibling:<12.2f} {hierarchical:<18.2f} {structural:<18.2f}")
        print("-" * 80)


if __name__ == "__main__":
    main()

