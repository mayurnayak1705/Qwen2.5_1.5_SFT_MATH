def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None

def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


import re
from typing import Union, List

def _extract_answer_candidates(s: str):
    """Return ordered candidate substrings to try for matching the ground truth."""
    s = (s or "").strip()
    candidates = []

    # 1) boxed LaTeX if present
    if "\\boxed" in s:
        try:
            boxed = extract_boxed_answer(s)
        except Exception:
            boxed = None
        if boxed:
            candidates.append(boxed)

    # 2) content inside $...$
    candidates += re.findall(r"\$(.*?)\$", s)

    # 3) LaTeX fragments: \frac, \sqrt
    candidates += re.findall(r"(\\frac\{.*?\}\{.*?\})", s)
    candidates += re.findall(r"(\\sqrt\{.*?\})", s)

    # 4) fractions like a/b, then floats/ints
    frac_tokens = re.findall(r"-?\d+\s*/\s*\d+", s)
    float_int_tokens = re.findall(r"-?\d+\.\d+|-?\d+", s)
    candidates += frac_tokens + float_int_tokens

    # 5) try text after "is" or "="
    m = re.search(r"(?:\b(?:is|=)\b)\s*(.+)$", s)
    if m:
        candidates.append(m.group(1).strip())

    # 6) fallback: whole string
    candidates.append(s)

    # remove dups while preserving order
    seen, out = set(), []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _equal_enough(a, b):
    """Check raw equality: try numeric, else string."""
    try:
        return float(a) == float(b)
    except Exception:
        return str(a).strip() == str(b).strip()


def score_model(model_output: str,
                actual_output: Union[str, float, int, List[str]],
                fast: bool = True,
                tolerant: bool = True,
                verbose: bool = True):
    """
    Score a model output against ground truth and print the result dictionary.

    Args:
      model_output: raw assistant output (ideally includes <answer>...</answer>)
      actual_output: ground truth (string, number, or list of acceptable strings)
      fast: passed through to grade(...); keep True for speed
      tolerant: if True, use candidate extraction and fallbacks
      verbose: if True, print diagnostic info

    Returns:
      dict with keys: format_reward, answer_reward, reward
    """
    # Normalize ground truth
    if isinstance(actual_output, (float, int)):
        gts = [str(actual_output)]
    elif isinstance(actual_output, list):
        gts = [str(gt) for gt in actual_output]
    else:
        gts = [str(actual_output)]

    # Extract block from <answer> ... </answer>
    model_answer_block = None
    if "</think> <answer>" in model_output and "</answer>" in model_output:
        model_answer_block = model_output.split("<answer>")[-1].replace("</answer>", "").strip()

    # If not found, try boxed-extraction fallback
    if model_answer_block is None:
        try:
            model_answer_block = extract_answer(model_output)
        except Exception:
            model_answer_block = None

    # Last fallback: whole output
    if model_answer_block is None:
        model_answer_block = model_output.strip()

    # Handle bad boxed
    if "\\boxed" in model_answer_block and extract_boxed_answer(model_answer_block) is None:
        result = {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
        if verbose:
            print("Result:", result)
            print("Reason: boxed construct present but no boxed content extracted.")
        return result

    # --- 1) Strict grading first ---
    for gt in gts:
        try:
            if grade(model_answer_block, gt, fast):
                result = {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
                if verbose:
                    print("Result:", result)
                    print("Matched (strict) block:", repr(model_answer_block), "vs", repr(gt))
                return result
        except Exception:
            pass

    if not tolerant:
        result = {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
        if verbose:
            print("Result:", result)
            print("Reason: strict mode only, no match.")
        return result

    # --- 2) Candidate extraction ---
    candidates = _extract_answer_candidates(model_answer_block)
    if verbose:
        print("Candidates to try:", candidates)

    for cand in candidates:
        for gt in gts:
            # Try grade() first
            try:
                if grade(cand, gt, fast):
                    result = {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
                    if verbose:
                        print("Result:", result)
                        print("Matched via grade:", repr(cand), "vs", repr(gt))
                    return result
            except Exception:
                pass

            # Fallback: direct numeric/string equality
            if _equal_enough(cand, gt):
                result = {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
                if verbose:
                    print("Result:", result)
                    print("Matched via fallback equality:", repr(cand), "vs", repr(gt))
                return result

    # --- 3) No match ---
    result = {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
    if verbose:
        print("Result:", result)
        print("No candidate matched.")
    return result

