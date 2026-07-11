"""
Plain-text prompt templates, section-marker parsers, and validation logic
for the robust A-MEM system. Replaces JSON-schema LLM calls with plain-text
prompts that work with any LLM backend (Ollama, SGLang, OpenAI, etc.).
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Callable
from nltk.stem import PorterStemmer

logger = logging.getLogger("amem_robust")
_STEMMER = PorterStemmer()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing.

    Many models emit valid JSON even without strict mode, so we try that first
    for best-of-both-worlds compatibility.
    """
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ---------------------------------------------------------------------------
# List parsing helpers
# ---------------------------------------------------------------------------

KEYWORD_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
    'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
    'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
    'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
}

KEYWORD_GENERIC_TERMS = {
    'activity', 'anything', 'chat', 'conversation', 'event', 'events',
    'experience', 'experiences', 'farewell', 'goodbye', 'happened', 'life',
    'meeting', 'people', 'person', 'session', 'situation', 'something',
    'stuff', 'talk', 'thing', 'things', 'time', 'topic', 'topics', 'work',
}

KEYWORD_CONVERSATION_FILLERS = {
    'awesome', 'cool', 'definitely', 'glad', 'great', 'hey', 'image', 'let',
    'like', 'looks', 'ok', 'okay', 'photo', 'really', 'sounds', 'thank',
    'thanks', 'wow', 'yeah', 'yep',
}

KEYWORD_TIME_TERMS = {
    'today', 'tomorrow', 'yesterday', 'tonight', 'morning', 'afternoon',
    'evening', 'night', 'week', 'weekend', 'month', 'year', 'daily',
}

KEYWORD_FILTER_TERMS = (
    KEYWORD_STOPWORDS
    | KEYWORD_GENERIC_TERMS
    | KEYWORD_TIME_TERMS
    | KEYWORD_CONVERSATION_FILLERS
)

KEYWORD_HARD_FILTER_TERMS = (
    KEYWORD_GENERIC_TERMS
    | KEYWORD_TIME_TERMS
    | KEYWORD_CONVERSATION_FILLERS
)

def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items.

    Handles:
      - Bullet points (-, *, numbered)
      - Comma-separated values
      - One item per line
    """
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        line = re.sub(r'^[\-\*\u2022]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        # If the line contains commas, split on them
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _keyword_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_keyword(keyword: Any) -> str:
    text = str(keyword).lower()
    text = re.sub(r"[_/-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def _light_stem(token: str) -> str:
    return _STEMMER.stem(token)


def _build_content_token_index(content: str) -> set[str]:
    tokens = set(_keyword_tokens(content))
    return tokens | {_light_stem(token) for token in tokens}


def _token_is_grounded(token: str, content_tokens: set[str]) -> bool:
    return token in content_tokens or _light_stem(token) in content_tokens


def _is_artifact_token(token: str) -> bool:
    return token.endswith("says")


# ---------------------------------------------------------------------------
# Keyword pruning mode
# ---------------------------------------------------------------------------

# Global pruning mode: "simple" (no stemming) or "nltk" (with PorterStemmer)
# Set via set_keyword_pruning_mode() or by importing and assigning directly.
_KEYWORD_PRUNING_MODE: str = "nltk"


def set_keyword_pruning_mode(mode: str) -> None:
    """Set the global keyword pruning mode.

    Args:
        mode: "none"   - no filtering; normalize + deduplicate only (raw LLM output)
              "simple" - stopword/generic/filler filtering + exact-match grounding (no stemming)
              "nltk"   - full KEA-style filtering with PorterStemmer derivational-variant matching
    """
    global _KEYWORD_PRUNING_MODE
    if mode not in ("none", "simple", "nltk"):
        raise ValueError(f"Unknown keyword pruning mode: {mode!r}. Use 'none', 'simple', or 'nltk'.")
    _KEYWORD_PRUNING_MODE = mode


def _token_is_grounded_simple(token: str, content_tokens: set) -> bool:
    """Grounding check WITHOUT stemming (exact token match only)."""
    return token in content_tokens


def sanitize_keywords_none(
    content: str,
    keywords: Any,
    max_keywords: int = 5,
) -> List[str]:
    """Pass-through pruning: normalize and deduplicate only, no filtering.

    This is the baseline/control condition: the raw LLM-generated keywords are
    kept as-is (modulo normalization and capping to max_keywords). No stopword
    removal, no grounding check, no stemming.
    """
    if isinstance(keywords, str):
        candidates = _parse_list_items(keywords)
    elif isinstance(keywords, list):
        candidates = keywords
    else:
        return []

    seen: set = set()
    result: List[str] = []
    for raw_keyword in candidates:
        keyword = _normalize_keyword(raw_keyword)
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        result.append(keyword)
        if len(result) >= max_keywords:
            break
    return result


def sanitize_keywords_simple(
    content: str,
    keywords: Any,
    max_keywords: int = 5,
) -> List[str]:
    """Prune noisy LLM keywords using simple rule filtering (no NLTK stemming).

    Applies the same stopword/generic/filler filtering and content-grounding
    check as sanitize_keywords(), but uses EXACT token matching for grounding
    instead of NLTK PorterStemmer derivational-variant matching.
    """
    if isinstance(keywords, str):
        candidates = _parse_list_items(keywords)
    elif isinstance(keywords, list):
        candidates = keywords
    else:
        return []

    # Build simple token index (no stemmed variants)
    content_tokens = set(_keyword_tokens(content or ""))
    content_lower = (content or "").lower()
    seen = set()
    filtered = []

    for raw_keyword in candidates:
        keyword = _normalize_keyword(raw_keyword)
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)

        tokens = _keyword_tokens(keyword)
        if any(_is_artifact_token(t) for t in tokens):
            continue
        if any(t in KEYWORD_HARD_FILTER_TERMS for t in tokens):
            continue
        meaningful_tokens = [t for t in tokens if t not in KEYWORD_FILTER_TERMS]
        if not meaningful_tokens:
            continue
        if len(meaningful_tokens) == 1 and meaningful_tokens[0] in KEYWORD_GENERIC_TERMS:
            continue
        # Exact-match grounding (no stemming)
        if content_tokens and not all(_token_is_grounded_simple(t, content_tokens) for t in meaningful_tokens):
            continue

        filtered.append((keyword, meaningful_tokens))

    kept = []
    for keyword, tokens in sorted(filtered, key=lambda item: (-len(item[1]), item[0])):
        token_set = set(tokens)
        if any(token_set < set(existing_tokens) for _, existing_tokens in kept):
            continue
        kept.append((keyword, tokens))

    def score(item):
        keyword, tokens = item
        frequency = sum(content_lower.count(token) for token in tokens)
        first_positions = [
            content_lower.find(token)
            for token in tokens
            if content_lower.find(token) >= 0
        ]
        first_pos = min(first_positions) if first_positions else len(content_lower) + 1
        phrase_bonus = min(len(tokens), 3)
        return (-frequency, -phrase_bonus, first_pos, keyword)

    ranked = sorted(kept, key=score)
    return [keyword for keyword, _ in ranked[:max_keywords]]


def sanitize_keywords(
    content: str,
    keywords: Any,
    max_keywords: int = 5,
) -> List[str]:
    """Dispatch to pruning implementation based on _KEYWORD_PRUNING_MODE.

    "none"   -> sanitize_keywords_none()   (normalize + dedup only, no filtering)
    "simple" -> sanitize_keywords_simple() (exact-match grounding, no stemming)
    "nltk"   -> _sanitize_keywords_nltk()  (PorterStemmer grounding)
    """
    if _KEYWORD_PRUNING_MODE == "none":
        return sanitize_keywords_none(content, keywords, max_keywords)
    if _KEYWORD_PRUNING_MODE == "simple":
        return sanitize_keywords_simple(content, keywords, max_keywords)
    return _sanitize_keywords_nltk(content, keywords, max_keywords)


def _sanitize_keywords_nltk(
    content: str,
    keywords: Any,
    max_keywords: int = 5,
) -> List[str]:
    """Prune noisy LLM keywords using KEA-style rule features.

    The rules are intentionally lightweight: normalize, remove generic terms,
    require grounding in the source content, prefer specific phrases,
    then keep the highest-scoring candidates.
    """
    if isinstance(keywords, str):
        candidates = _parse_list_items(keywords)
    elif isinstance(keywords, list):
        candidates = keywords
    else:
        return []

    content_tokens = _build_content_token_index(content or "")
    content_lower = (content or "").lower()
    seen = set()
    filtered = []

    for raw_keyword in candidates:
        keyword = _normalize_keyword(raw_keyword)
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)

        tokens = _keyword_tokens(keyword)
        if any(_is_artifact_token(t) for t in tokens):
            continue
        if any(t in KEYWORD_HARD_FILTER_TERMS for t in tokens):
            continue
        meaningful_tokens = [t for t in tokens if t not in KEYWORD_FILTER_TERMS]
        if not meaningful_tokens:
            continue
        if len(meaningful_tokens) == 1 and meaningful_tokens[0] in KEYWORD_GENERIC_TERMS:
            continue
        if content_tokens and not all(_token_is_grounded(t, content_tokens) for t in meaningful_tokens):
            continue

        filtered.append((keyword, meaningful_tokens))

    kept = []
    for keyword, tokens in sorted(filtered, key=lambda item: (-len(item[1]), item[0])):
        token_set = set(tokens)
        if any(token_set < set(existing_tokens) for _, existing_tokens in kept):
            continue
        kept.append((keyword, tokens))

    def score(item):
        keyword, tokens = item
        frequency = sum(content_lower.count(token) for token in tokens)
        first_positions = [
            content_lower.find(token)
            for token in tokens
            if content_lower.find(token) >= 0
        ]
        first_pos = min(first_positions) if first_positions else len(content_lower) + 1
        phrase_bonus = min(len(tokens), 3)
        return (-frequency, -phrase_bonus, first_pos, keyword)

    ranked = sorted(kept, key=score)
    return [keyword for keyword, _ in ranked[:max_keywords]]


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end).

    Args:
        text: Full LLM response
        marker: Section header to find (e.g. "KEYWORDS")
        next_markers: List of possible next section headers

    Returns:
        The text content of that section (may be empty string).
    """
    # Build a regex that finds the marker (case-insensitive) followed by a colon
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    # The first line of content may be on the same line as the marker
    first_line = match.group(1).strip()

    # Find where the next section starts
    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


# ---------------------------------------------------------------------------
# Parsers for each call site
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the analyze_content LLM response.

    Returns:
        {"keywords": [...], "context": "...", "tags": [...]}
    """
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context = context_text.strip() if context_text.strip() else ""
        tags = _parse_list_items(tags_text)

        return {"keywords": keywords, "context": context, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse, content)

    # Validate / repair
    result = validate_analysis_result(result, content)
    return result


def parse_evolution_decision(response: str) -> Dict[str, str]:
    """Parse the evolution decision response.

    Returns:
        {"decision": "NO_EVOLUTION|STRENGTHEN|UPDATE_NEIGHBOR|STRENGTHEN_AND_UPDATE",
         "reason": "..."}
    """
    def _section_parse(resp: str) -> Dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        # Normalize common variants
        valid_decisions = {
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE"
        }
        if decision not in valid_decisions:
            # Try to infer from keywords
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    # Map JSON keys if we got JSON
    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", [])
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    if "decision" not in result:
        result = {"decision": "NO_EVOLUTION", "reason": ""}

    return result


def parse_strengthen_details(response: str) -> Dict[str, Any]:
    """Parse the strengthen details response.

    Returns:
        {"connections": [int, ...], "tags": [str, ...]}
    """
    def _section_parse(resp: str) -> Dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS", ["CONNECTIONS"])

        # Parse connections as integers
        connections = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass

        tags = _parse_list_items(tags_text)
        return {"connections": connections, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse)

    # Map from JSON keys if needed
    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [int(x) for x in result.get("suggested_connections", []) if isinstance(x, (int, float))]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(response: str, num_neighbors: int) -> List[Dict[str, Any]]:
    """Parse the update neighbors response.

    Returns:
        [{"context": "...", "tags": [...]}, ...] — one per neighbor
    """
    def _section_parse(resp: str, n_neighbors: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i in range(n_neighbors):
            # Try to find NEIGHBOR i: block
            # Look for "NEIGHBOR i:" or "NEIGHBOR i\n"
            pattern = re.compile(
                rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE
            )
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            # Find the end of this neighbor block (next NEIGHBOR or end)
            next_pattern = re.compile(
                rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE
            )
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end():block_end]

            ctx = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS", ["CONTEXT"])
            tags = _parse_list_items(tags_text)

            neighbors.append({"context": ctx.strip(), "tags": tags})

        return neighbors

    # Try JSON first
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            neighbors = []
            for i in range(num_neighbors):
                ctx = contexts[i] if i < len(contexts) else ""
                tags = tags_list[i] if i < len(tags_list) else []
                neighbors.append({"context": ctx, "tags": tags})
            return neighbors
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


def parse_plain_text_answer(response: str) -> str:
    """Parse a plain-text answer response (for QA evaluation).

    If the model returned JSON with an "answer" field, extract it.
    Otherwise return the raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_relevant_parts(response: str) -> str:
    """Parse retrieve_memory_llm response.

    If JSON with "relevant_parts", extract it. Otherwise return raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "relevant_parts" in data:
            return str(data["relevant_parts"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_keywords_response(response: str) -> str:
    """Parse generate_query_llm response.

    If JSON with "keywords", extract it. Otherwise return raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


# ---------------------------------------------------------------------------
# Validation / heuristic repair
# ---------------------------------------------------------------------------

def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Validate and repair the analysis result.

    - If keywords is empty, extract capitalized words / nouns heuristically.
    - If context is empty, use the first sentence of content.
    - If tags is empty, derive from keywords.
    """
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    # Ensure lists
    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    # Repair empty keywords from content
    if not keywords and content:
        keywords = _heuristic_keywords(content)

    keywords = sanitize_keywords(content, keywords)
    if not keywords and content:
        keywords = sanitize_keywords(content, _heuristic_keywords(content))

    # Repair empty context from content
    if not context and content:
        context = _heuristic_context(content)

    # Repair empty tags from keywords
    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    # Prefer capitalized words (likely proper nouns / key terms)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in KEYWORD_FILTER_TERMS or _is_artifact_token(w_lower) or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))

    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Extract a heuristic context sentence from content."""
    # Take the first sentence (up to period, question mark, or exclamation)
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    # Fallback: first 200 chars
    return content[:200].strip()
