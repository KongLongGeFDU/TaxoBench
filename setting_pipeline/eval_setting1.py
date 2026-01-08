from openai import OpenAI
from anthropic import Anthropic
import os
import time
import re
import json
import argparse
import multiprocessing
from functools import partial
from tqdm import tqdm  # Recommended: pip install tqdm

# Basic JSON read
def read_json_basic(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(path):
    """Reads a jsonl file and returns a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def append_to_jsonl(json_data, jsonl_path):
    """
    Appends a JSON object to a jsonl file.
    Each json_data is written as a separate line.
    """
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False))
            f.write("\n")
            
            # === Core Modification ===
            # 1. Force push Python internal buffer to OS buffer
            f.flush()
            # 2. Force OS to write to physical disk immediately 
            # (Prevents data loss during power failure or process crash)
            os.fsync(f.fileno())
            # =========================
            
    except Exception as e:
        print(f"❌ Write Error: {e}", flush=True)

def use_gpt(contents, thinking, model, temperature=1):
    # reasoning_effort="none (default), low, medium, high"
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"), # Set env var or fill here
        api_key=os.getenv("OPENAI_API_KEY")    # Set env var or fill here
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
        base_url=os.getenv("GEMINI_BASE_URL"), 
        api_key=os.getenv("GEMINI_API_KEY")
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
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
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
        base_url=os.getenv("QWEN_BASE_URL"),
        api_key=os.getenv("QWEN_API_KEY"),
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
        base_url=os.getenv("KIMI_BASE_URL"),
        api_key=os.getenv("KIMI_API_KEY"),
        timeout=1800  # Set timeout to 300s (comment says 300, code says 1800)
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
        base_url=os.getenv("CLAUDE_BASE_URL"),
        api_key=os.getenv("CLAUDE_API_KEY"),
    )
    if thinking:
        # thinking={"type": "enabled", "budget_tokens": 10000}, default is disabled
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
    contents: list of messages
    model: Optional, required for providers like deepseek
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

Your task is to organize a set of research papers into a **hierarchical topic tree**
(bottom-up), given:
- a survey topic,
- for each paper: title, abstract, and an structured summary
  (which may include research problem, motivation, methodology, and findings).

Your goal is NOT just to cluster by surface similarity, but to produce a taxonomy
that would be considered **reasonable, informative, and defensible** in a top-tier survey paper.

### Hard Constraints
1. Output must be **strictly valid JSON**.
2. Only leaf nodes may contain `"papers"`; all internal nodes must contain `"subtopics"`.
3. **Every paper must appear exactly once** in the entire tree.
4. NO duplicate papers anywhere.
5. The tree must eventually merge into **one single root node**.

### CLASSIFICATION RULES
- Group papers by semantic similarity using both title + abstract + summary.
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
    Converts raw quotes " inside JSON values to escaped quotes \".
    Assumptions:
    - Keys are valid.
    - Structure is { "key": "value" }.
    - Errors originate only from within the value string.
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
                # Determine if this is the end of the value string.
                # If the following character is not , or }, it's likely a raw quote inside the value.
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
    
    # Safety check: if data is not a list, return empty string
    if not isinstance(data, list):
        return ""

    for i, paper_data in enumerate(data, 1):
        if not isinstance(paper_data, dict):
            continue

        # --- 1. Get Title ---
        title = paper_data.get("title", "No Title")

        # --- 2. Get Abstract (JSON key is "abs") ---
        abstract = paper_data.get("abs", "No Abstract")

        # --- 3. Process Core Info (Previously Contributions Logic) ---
        key_elements_list = []
        summary = paper_data.get("summary")
        
        if summary and isinstance(summary, dict):
            # Note: Even though we changed the variable names, the key in source JSON might still be "contributions".
            # Keep as is unless source data keys change.
            contrib_data = summary.get("contributions", [])
            
            for item in contrib_data:
                name = item.get("name", "")
                desc = item.get("description", "")
                # Format
                key_elements_list.append(f"- {name}: {desc}")
        
        if key_elements_list:
            details_block = "\n    ".join(key_elements_list)
        else:
            details_block = "N/A"

        # --- 4. Assemble Single Paper Info ---
        paper_section = (
            f"Paper {i}:\n"
            f"  Title: {title}\n"
            f"  Abstract: {abstract}\n"
            f"  Structured Summary:\n" 
            f"    {details_block}\n"
        )
        
        prompt_parts.append(paper_section)

    return "\n".join(prompt_parts)

# #NOTE: Multiprocessing
# Global variable to hold lock in child processes
file_lock = None

def init_lock(l):
    """
    Process initializer: ensures every child process has access to the same lock.
    """
    global file_lock
    file_lock = l

def safe_append_to_jsonl(item, path):
    """
    Locked file write function to prevent data corruption from concurrent writes.
    """
    global file_lock
    # Only the process holding the lock can write
    with file_lock:
        append_to_jsonl(item, path)

def process_single_item(item, args):
    """
    Logic for processing a single item (originally the eval loop body).
    """
    # Extract args
    eval_path = args.eval_path
    provider = args.provider
    model = args.model
    temperature = args.temperature
    thinking = args.thinking
    
    # -------------------------------------------------
    # Business logic
    # -------------------------------------------------
    print(f"====Processing item {item['id']} ｜ {item.get('survey_topic', 'N/A')}====")
    
    survey_name = item.get("survey_topic", "N/A")
    paper_list = item.get("pdfs", [])
    papers_prompt = build_papers_prompt(paper_list)
    paper_num = len(paper_list)

    item["input_paper_count"] = paper_num
    
    # Construct Prompt
    contents = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt.format(num_paper=paper_num, survey_name=survey_name, papers=papers_prompt)}
    ]

    item["input_content"] = f"SYSTEM PROMPT:\n{contents[0]['content']}\n\nUSER PROMPT:\n{contents[1]['content']}"

    try:
        start_time = time.time()
        # === Core Prevention: API Rate Limits ===
        # Recommendation: If concurrency is high, add a small random sleep here to avoid hitting limits.
        # time.sleep(random.uniform(0.1, 1.0)) 
        
        response = call_llm(provider, contents, thinking, model=model, temperature=temperature)
        execution_time = time.time() - start_time
    except Exception as e:
        print(f"Error processing item {item['id']}: {e}")
        item["hierarchy_tree"] = "FIX FAILED"
        # Write even on failure to prevent data loss
        safe_append_to_jsonl(item, eval_path)
        return

    # Parsing Logic
    pattern = r"```json\s*(.*?)```"
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
                    fixed_str = safe_json_loads(json_str)
                    item["hierarchy_response"] = fixed_str
                    json_data = json.loads(fixed_str)
                except json.JSONDecodeError as e:
                    # Attempt Repair
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
            print(f"match error===={e}")
            item["hierarchy_response"] = response
            item["hierarchy_tree"] = "FIX FAILED"

    # === Key Point: Locked Write ===
    safe_append_to_jsonl(item, eval_path)

def run_parallel_eval(args):
    path = args.path
    # 1. Load data
    all_data = load_jsonl(path)
    
    # 2. Create a Manager Lock
    lock = multiprocessing.Lock()
    
    # 3. Configure Concurrency
    # If num_workers is not provided, default to CPU count - 2 to prevent freezing
    max_workers = args.num_workers if args.num_workers else max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting parallel processing with {max_workers} workers...")

    # 4. Use functools.partial to fix args, keeping item as variable
    process_func = partial(process_single_item, args=args)

    # 5. Start Process Pool
    # initializer=init_lock, initargs=(lock,) ensures the lock is passed to children
    with multiprocessing.Pool(processes=max_workers, initializer=init_lock, initargs=(lock,)) as pool:
        # Use tqdm for progress bar
        # imap_unordered is more efficient than map as it yields results as they finish
        list(tqdm(pool.imap_unordered(process_func, all_data), total=len(all_data), desc="Processing"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    
    # New arg: Control process count
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes")

    args = parser.parse_args()

    # Run Parallel Eval
    run_parallel_eval(args)
    print("Evaluation completed.")