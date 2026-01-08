from openai import OpenAI
from anthropic import Anthropic
import os
import time
import re
import json
import argparse
import multiprocessing
from multiprocessing import Manager
from functools import partial
from tqdm import tqdm

def read_json_basic(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(path):
    """Read jsonl file and return a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def append_to_jsonl(json_data, jsonl_path):
    """
    Append a JSON object to a jsonl file.
    """
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False))
            f.write("\n")
            
            # === Core Modification ===
            # 1. Flush Python internal buffer to OS buffer
            f.flush()
            # 2. Force OS to write buffer to physical disk (prevents data loss on crash)
            os.fsync(f.fileno())
            
    except Exception as e:
        print(f"❌ Write Error: {e}", flush=True)

def use_gpt(contents, thinking, model, temperature=1):
    client = OpenAI(
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY"
    )
    if thinking:
        print("**********************Thinking**********************")
        completion = client.chat.completions.create(
            model=model,
            messages=contents,
            temperature=temperature,
            reasoning_effort="high"
        )
    else:
        print("**********************No-Thinking**********************")
        completion = client.chat.completions.create(
            model=model,
            messages=contents,
            temperature=temperature,
        )
    response = completion.choices[0].message.content
    return response

def use_gemini(contents, thinking, model, temperature=1):
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="YOUR_BASE_URL"
    )
    if thinking:
        print("**********************Thinking**********************")
        response = client.chat.completions.create(
            model=model,
            messages=contents,
            temperature=temperature
        )
    else:
        print("**********************No-Thinking**********************")
        response = client.chat.completions.create(
            model=model,
            messages=contents,
            temperature=temperature,
            reasoning_effort="low"
        )
    return response.choices[0].message.content

def use_deepseek(contents, thinking, model, temperature=1):
    client = OpenAI(
        # OpenAI-compatible SDKs usually require the /v1 suffix
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY",
    )
    if thinking:
        print("**********************Thinking**********************")
        chat_completion = client.chat.completions.create(
            messages=contents,
            model=model,
            temperature=temperature,
            extra_body={"thinking": {"type": "enabled"}}
        )
    else:
        print("**********************No-Thinking**********************")
        chat_completion = client.chat.completions.create(
            messages=contents,
            model=model,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}}
        )
    response = chat_completion.choices[0].message.content
    return response

def use_qwen(contents, thinking, model, temperature=1):
    client = OpenAI(
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY",
    )
    if thinking:
        print("**********************Thinking**********************")
        chat_completion = client.chat.completions.create(
            messages=contents,
            model=model,
            temperature=temperature,
            extra_body={"thinking": {"type": "enabled"}}
        )
    else:
        print("**********************No-Thinking**********************")
        chat_completion = client.chat.completions.create(
            messages=contents,
            model=model,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}}
        )
    response = chat_completion.choices[0].message.content
    return response

def use_kimi(contents, thinking, model, temperature=1):
    client = OpenAI(
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY",
        timeout=1800 
    )
    if thinking:
        print("**********************Thinking**********************")
        chat_completion = client.chat.completions.create(
            messages=contents,
            model=model,
            temperature=temperature,
            extra_body={"thinking": {"type": "enabled"}}
        )
    else:
        print("**********************No-Thinking**********************")
        chat_completion = client.chat.completions.create(
            messages=contents,
            model=model,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}}
        )
    response = chat_completion.choices[0].message.content
    return response

def use_claude(user_content, thinking, model, system_content=None, temperature=1):
    client = Anthropic(
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY",
    )
    if thinking:
        print("**********************Thinking**********************")
        temperature = max(temperature, 1)
        response = client.messages.create(
            model=model,
            max_tokens=18000,
            system=system_content,
            messages=user_content,
            temperature=temperature,
            thinking={"type": "enabled", "budget_tokens": 12000}
        )
        return response.content[1].text

    else:
        print("**********************No-Thinking**********************")
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_content,
            messages=user_content,
            temperature=temperature,
        )
        return response.content[0].text 

def call_llm(provider, contents, thinking, model=None, temperature=1):
    """
    provider: 'gpt' | 'gemini' | 'deepseek' | 'claude' | 'qwen' | 'kimi'
    """
    provider = provider.lower()

    if provider == "gpt":
        return use_gpt(contents, thinking, model=model, temperature=temperature)

    elif provider == "gemini":
        return use_gemini(contents, thinking, model=model, temperature=temperature)

    elif provider == "deepseek":
        return use_deepseek(contents, thinking, model=model, temperature=temperature)
    
    elif provider == "qwen":
        return use_qwen(contents, thinking, model=model, temperature=temperature)
    
    elif provider == "kimi":
        if thinking:
            return use_kimi(contents, thinking, model="Kimi-K2-thinking", temperature=temperature)
        return use_kimi(contents, thinking, model="Kimi-K2-0905", temperature=temperature)

    elif provider == "claude":
        system_content = [msg for msg in contents if msg["role"] == "system"]
        user_content = [msg for msg in contents if msg["role"] == "user"]
        for msg in system_content:
            msg["type"] = "text"
            msg["text"] = msg["content"]
            msg["cache_control"] = {"type": "ephemeral"}
            del msg["role"]
            del msg["content"]
        
        return use_claude(user_content, thinking, model=model, system_content=system_content, temperature=temperature)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


system_prompt = """You are a senior researcher and survey-author with deep experience in structuring 
high-quality academic survey papers.

Your task is to organize a set of research papers into a **hierarchical topic tree** (bottom-up), given:
- a survey topic,
- for each paper: title, abstract, extracted Core Tasks (the primary problem addressed), 
  and Contributions (key innovations).

Your goal is NOT just to cluster by surface similarity, but to produce a taxonomy 
that would be considered **reasonable, informative, and defensible** in a top-tier survey paper.

### Hard Constraints
1. Output must be **strictly valid JSON**.
2. Only leaf nodes may contain `"papers"`; all internal nodes must contain `"subtopics"`.
3. **Every paper must appear exactly once** in the entire tree.
4. NO duplicate papers anywhere.
5. The tree must eventually merge into **one single root node**.

### CLASSIFICATION RULES
- Group papers by semantic similarity using both title + abstract + Core Task + Contributions.
- Create meaningful names for leaf-level themes.

### ANTI-DUPLICATION PROCEDURE (MANDATORY)
Before constructing the tree:
1. Produce an internal list of all given paper titles.
2. Assign each paper to exactly one leaf node.
3. After assignment, verify that:
   - the number of assigned papers equals the number of input papers,
   - no paper appears in more than one group.

### Output Format
Use a JSON structure like this (replace placeholders with actual paper titles) and
the output you produce MUST be wrapped inside a fenced code block:

```json
{
  "name": "AI Research",
  "subtopics": [
    {
      "name": "NLP",
      "subtopics": [
        {
          "name": "Text Summarization",
          "papers": ["<actual paper titles>"]
        },
        {
          "name": "Machine Translation",
          "papers": ["<actual paper titles>"]
        }
      ]
    }
  ]
}```"""

user_prompt = """Perform a bottom-up hierarchical clustering of the following {num_paper} papers and produce a JSON research topic tree.

Survey Topic: {survey_name}

Paper List:
{papers}

Before returning, check that:
- Every paper title appears exactly once.
- Only leaf nodes have a "papers" field.
- All intermediate nodes have a "subtopics" field.
- The JSON is strictly valid and parsable.

### Output:"""

fixed_system_prompt = """You are a strict JSON repair tool."""

fixed_user_prompt = """"Your task is to repair the following JSON text so that it becomes valid, strictly parsable JSON.

Rules you MUST follow:
1. The output MUST be valid JSON and parsable by standard JSON parsers.
2. Do NOT change the overall structure, hierarchy, or ordering of keys.
3. Do NOT rename keys or merge/split objects or arrays.
4. Fix ONLY syntax-level errors, including but not limited to:
   - Unescaped or mismatched quotation marks
   - Missing or extra commas
   - Invalid escape sequences
   - Mismatched brackets or braces
5. If a value is irreparably broken, you may remove ONLY that specific key-value pair.
6. Preserve all original text content and semantics as much as possible.
7. Do NOT add explanations, comments, or markdown.
8. Output ONLY the repaired JSON, nothing else.

Here is the invalid JSON:

{json_str}
"""

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        fixed = escape_quotes_inside_json_values(s)
        return json.loads(fixed)

def escape_quotes_inside_json_values(s: str) -> str:
    """
    Convert raw quotes " inside JSON values to \".
    Assumptions:
    - Keys are valid.
    - Structure is { "key": "value" }.
    - Errors only occur inside values.
    """

    out = []
    in_string = False
    escaped = False
    in_value_string = False

    for i, ch in enumerate(s):
        if ch == '\\' and not escaped:
            escaped = True
            out.append(ch)
            continue

        if ch == '"' and not escaped:
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                # Check if it is the end of the value string
                # If next char is not , or }, it is likely a raw quote inside value
                lookahead = s[i+1:].lstrip()
                if in_value_string and lookahead and lookahead[0] not in [',', '}']:
                    out.append(r'\"')
                else:
                    in_string = False
                    in_value_string = False
                    out.append(ch)
            escaped = False
            continue

        if ch == ':' and not in_string:
            # Next string is a value
            in_value_string = True
            out.append(ch)
            continue

        escaped = False
        out.append(ch)

    return ''.join(out)

def build_papers_prompt(data):
    prompt_parts = []
    
    if not isinstance(data, list):
        return ""

    for i, paper_data in enumerate(data, 1):
        if not isinstance(paper_data, dict):
            continue

        # --- 1. Get Title ---
        title = paper_data.get("title", "No Title")

        # --- 2. Get Abstract ---
        abstract = paper_data.get("abs", "No Abstract")

        # --- 3. Get Core Task and Contributions ---
        summary = paper_data.get("summary", {})
        
        # 3.1 Extract Core Task
        core_task_text = "N/A"
        if summary and isinstance(summary, dict):
            core_task_text = summary.get("core_task", {}).get("text", "N/A")

        # 3.2 Extract Contributions List
        contrib_lines = []
        if summary and isinstance(summary, dict):
            contrib_data = summary.get("contributions", [])
            for item in contrib_data:
                name = item.get("name", "")
                desc = item.get("description", "")
                contrib_lines.append(f"- {name}: {desc}")
        
        # Handle indentation
        if contrib_lines:
            # Handle indentation for subsequent lines
            contrib_block = "\n    ".join(contrib_lines)
        else:
            contrib_block = "N/A"

        # --- 4. Assemble Single Paper Info ---
        paper_section = (
            f"Paper {i}:\n"
            f"  Title: {title}\n"
            f"  Abstract: {abstract}\n"
            f"  Core Task: {core_task_text}\n"
            f"  Contributions:\n"
            f"    {contrib_block}\n"  # Indentation for the first line
        )
        
        prompt_parts.append(paper_section)

    return "\n".join(prompt_parts)

# Note: Multiprocessing logic
def process_single_item(item, args, file_lock):
    """
    Process a single item.
    Args and file_lock are fixed via partial.
    """
    path = args.path
    eval_path = args.eval_path
    provider = args.provider
    model = args.model
    temperature = args.temperature
    thinking = args.thinking
    
    pattern = r"```json\s*(.*?)```"

    try:
        # Get survey topic
        survey_name = item.get("survey_topic", "N/A")

        # Get title, abstract, summary
        paper_list = item.get("pdfs", [])
        # Ensure no global variable dependency inside this function
        papers_prompt = build_papers_prompt(paper_list) 
        paper_num = len(paper_list)

        item["input_paper_count"] = paper_num
        
        contents = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(num_paper=paper_num, survey_name=survey_name, papers=papers_prompt)}
        ]

        item["input_content"] = f"SYSTEM PROMPT:\n{contents[0]['content']}\n\nUSER PROMPT:\n{contents[1]['content']}"
        print(item["input_content"])
        
        # === Call LLM (Time-consuming operation) ===
        start_time = time.time()
        response = call_llm(provider, contents, thinking, model=model, temperature=temperature)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # === Parse Result ===
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                try:
                    json_data = json.loads(json_str)
                    item["hierarchy_response"] = json_str
                    item["hierarchy_tree"] = json_data
                except json.JSONDecodeError:
                    try:
                        # Attempt simple fix
                        fixed_str = safe_json_loads(json_str)
                        item["hierarchy_response"] = fixed_str
                        json_data = json.loads(fixed_str)
                    except json.JSONDecodeError as e:
                        # Attempt repair via LLM
                        formatted_user_prompt = fixed_user_prompt.format(json_str=json_str)
                        fixed_messages = [{"role": "user", "content": formatted_user_prompt}]
                        fixed_response = use_gpt(contents=fixed_messages, thinking=False, model="gpt-5", temperature=0)
                        fixed_json_data = json.loads(fixed_response)
                        item["hierarchy_response"] = fixed_response
                        item["hierarchy_tree"] = fixed_json_data
            except Exception as e:
                print(f"fixed error====: {e}")
                item["hierarchy_tree"] = "FIX FAILED"
        else:
            try:
                json_data = json.loads(response)
                item["hierarchy_tree"] = json_data
            except json.JSONDecodeError as e:
                print(f"math error===={e}")
                item["hierarchy_response"] = response
                item["hierarchy_tree"] = "FIX FAILED"

        # === Critical: Write to file (Locking) ===
        # Lock only during file writing to maximize parallel efficiency
        with file_lock:
            append_to_jsonl(item, eval_path)
            
        return True, item['id'] 

    except Exception as e:
        # Catch all exceptions to prevent process crash on single item error
        print(f"❌ Error processing item {item.get('id', 'unknown')}: {e}", flush=True)
        item["hierarchy_tree"] = "FIX FAILED"
        with file_lock:
            append_to_jsonl(item, eval_path)
        return False, item.get('id', 'unknown')


# ==========================================
# Main Program
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    
    # Process control
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel processes")
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.path}...")
    all_data = load_jsonl(args.path)
    filtered_data = all_data 
    print(f"Total items to process: {len(filtered_data)}")

    # 2. Prepare Multiprocessing Environment
    # Manager().Lock() can be shared across multiprocessing Pools
    manager = Manager()
    file_lock = manager.Lock()

    # Use partial to fix args and lock, so map only receives 'item'
    process_func = partial(process_single_item, args=args, file_lock=file_lock)

    # 3. Start Process Pool
    num_workers = args.num_processes
    print(f"Starting execution with {num_workers} processes...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered allows results to yield as soon as they are ready
        results = list(tqdm(
            pool.imap_unordered(process_func, filtered_data),
            total=len(filtered_data),
            desc="Processing"
        ))

    print("\nEvaluation completed.")

if __name__ == "__main__":
    main()