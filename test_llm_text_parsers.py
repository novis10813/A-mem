from llm_text_parsers import _heuristic_keywords, sanitize_keywords, validate_analysis_result


def test_sanitize_keywords_removes_generic_time_and_ungrounded_terms():
    content = (
        "Alice discussed the Mars strategy for the project launch. "
        "The Mars strategy was urgent."
    )
    keywords = [
        "Project",
        "thing",
        "Mars strategy",
        "quantum finance",
        "project launch",
        "launch",
        "today",
    ]

    assert sanitize_keywords(content, keywords) == ["mars strategy", "project launch"]


def test_sanitize_keywords_removes_partially_hallucinated_phrases():
    content = "Alice discussed the Mars strategy for the project launch."
    keywords = ["mars quantum finance", "mars strategy"]

    assert sanitize_keywords(content, keywords) == ["mars strategy"]


def test_sanitize_keywords_does_not_ground_by_fuzzy_prefix():
    content = "The team planned the application deployment."
    keywords = ["apple deployment", "application deployment"]

    assert sanitize_keywords(content, keywords) == ["application deployment"]


def test_sanitize_keywords_normalizes_deduplicates_and_prefers_phrase():
    content = "John enjoyed an RPG game with deep exploration."
    keywords = ["RPG game!", "rpg game", "game", "conversation"]

    assert sanitize_keywords(content, keywords) == ["rpg game"]


def test_sanitize_keywords_removes_locomo_speaker_artifacts():
    content = "Speaker Carolinesays : Caroline joined a charity race."
    keywords = ["carolinesays", "Caroline", "charity race", "race"]

    assert sanitize_keywords(content, keywords) == ["charity race", "caroline"]


def test_sanitize_keywords_removes_conversation_fillers_but_keeps_topics():
    content = "Wow, thanks! The charity race and photography project helped."
    keywords = ["wow", "thanks", "cool", "charity race", "photography", "project"]

    assert sanitize_keywords(content, keywords) == ["charity race", "photography", "project"]


def test_sanitize_keywords_rejects_phrases_containing_hard_artifacts():
    content = "The project image helped. Thanks, the project shipped."
    keywords = ["project image", "thanks project", "project"]

    assert sanitize_keywords(content, keywords) == ["project"]


def test_sanitize_keywords_rejects_phrases_containing_generic_filtered_tokens():
    content = "Mars strategy was urgent."
    keywords = ["mars conversation", "mars strategy"]

    assert sanitize_keywords(content, keywords) == ["mars strategy"]


def test_sanitize_keywords_drops_phrase_with_artifact_token():
    content = "Speaker Melaniesays : Melanie painted a lake sunrise."
    keywords = ["melaniesays lake", "lake sunrise"]

    assert sanitize_keywords(content, keywords) == ["lake sunrise"]


def test_sanitize_keywords_grounds_derivational_variants_with_nltk_stemming():
    content = (
        "She stayed motivated, felt excited, got creative, "
        "and really appreciated the support."
    )
    keywords = ["motivation", "excitement", "creativity", "appreciation"]

    assert sanitize_keywords(content, keywords) == [
        "appreciation",
        "creativity",
        "excitement",
        "motivation",
    ]


def test_sanitize_keywords_keeps_artifact_filters_with_nltk_stemming():
    content = "Speaker Johnsays : Wow, thanks for the motivating photo."
    keywords = ["johnsays", "wow", "thanks", "photo", "motivation"]

    assert sanitize_keywords(content, keywords) == ["motivation"]


def test_sanitize_keywords_still_removes_partially_hallucinated_stemmed_phrases():
    content = "She stayed motivated after the project launch."
    keywords = ["motivation quantum finance", "project launch"]

    assert sanitize_keywords(content, keywords) == ["project launch"]


def test_sanitize_keywords_caps_to_top_five_grounded_terms():
    content = "Alpha beta gamma delta epsilon zeta all appear in this note."
    keywords = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

    result = sanitize_keywords(content, keywords)

    assert len(result) == 5
    assert set(result).issubset(set(keywords))


def test_validate_analysis_result_sanitizes_llm_keywords():
    content = "Speaker Alice says: I adopted a cat named Luna."
    result = validate_analysis_result(
        {
            "keywords": ["conversation", "today", "quantum finance", "cat"],
            "context": "Pet adoption conversation",
            "tags": [],
        },
        content,
    )

    assert result["keywords"] == ["cat"]
    assert result["tags"] == ["cat"]


def test_validate_analysis_result_falls_back_when_all_keywords_are_pruned():
    content = "Speaker Alice says: I adopted a cat named Luna."
    result = validate_analysis_result(
        {
            "keywords": ["conversation", "today", "quantum finance"],
            "context": "Pet adoption conversation",
            "tags": [],
        },
        content,
    )

    assert result["keywords"]
    assert "cat" in result["keywords"]
    assert "conversation" not in result["keywords"]


def test_heuristic_keywords_excludes_speaker_artifact_tokens():
    content = "Speaker Carolinesays : Caroline joined a charity race."

    result = _heuristic_keywords(content)

    assert "carolinesays" not in result
