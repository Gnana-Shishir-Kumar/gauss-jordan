from research_agent.evidence import (
    Candidate, assess, best_excerpt, content_hash, domain_of, normalize_url, overlap,
)


def test_normalize_url():
    assert normalize_url("HTTPS://WWW.Example.com/a/b/?utm_source=x&b=2&a=1#frag") == "https://example.com/a/b?a=1&b=2"
    assert normalize_url("https://example.com") == "https://example.com/"
    assert normalize_url("https://example.com:443/x") == "https://example.com/x"
    assert normalize_url("https://example.com/x?fbclid=1") == "https://example.com/x"
    assert domain_of("https://www.Sub.Example.com/x") == "sub.example.com"


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("a") != content_hash("b")


def test_best_excerpt_picks_relevant_window_and_respects_limit():
    filler = "\n".join(f"Unrelated paragraph number {i} about cooking pasta." for i in range(60))
    target = "Solar panel efficiency reached 47 percent in tandem perovskite cells."
    text = filler + "\n" + target + "\n" + filler
    ex = best_excerpt(text, "solar panel efficiency perovskite", max_chars=300)
    assert target in ex and len(ex) <= 300


def _c(url, content, score=0.9, sq=1):
    return Candidate(sq, url, "t", None, content, score, True)


def test_assess_dedupes_and_records_reasons():
    seen_u, seen_h = set(), set()
    cands = [
        _c("https://a.com/1", "solar efficiency record"),
        _c("https://a.com/1", "solar efficiency other text"),          # duplicate url
        _c("https://b.com/1", "Solar  efficiency   RECORD"),            # duplicate content
        _c("https://c.com/1", ""),                                      # empty
        _c("https://d.com/1", "cooking pasta recipes", score=0.0),      # low relevance
    ]
    ds = assess(cands, "solar efficiency", seen_u, seen_h, min_relevance=0.15, excerpt_chars=500)
    assert [d.drop_reason for d in ds] == [None, "duplicate_url", "duplicate_content", "empty_content", "low_relevance"]
    assert [d.kept for d in ds] == [True, False, False, False, False]
    assert seen_u == {"https://a.com/1"}


def test_overlap():
    assert overlap("solar efficiency", "the solar record") == 0.5
    assert overlap("the and", "anything") == 0.0
