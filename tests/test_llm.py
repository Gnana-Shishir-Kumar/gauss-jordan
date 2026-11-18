"""TokenFactoryLLM against a mocked AsyncOpenAI client."""

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from research_agent.config import Network
from research_agent.backends.llm import LLMOutputError, TokenFactoryLLM, extract_json_text
from research_agent.schemas import Draft, Plan
from research_agent.backends.tavily import TavilyBackend

REQ = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")
NET = Network(max_retries=3, backoff_base_s=0, backoff_max_s=0, llm_timeout_s=2)


def resp(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.kwargs = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return resp(item)


def make(script):
    comp = FakeCompletions(script)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return TokenFactoryLLM(client, NET), comp


def call(llm, schema, context=None):
    return asyncio.run(llm.complete_json(model="m", system="s", user="u", schema=schema, context=context))


def test_extract_json_text():
    assert extract_json_text('<think>hmm</think>```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_text('Sure! {"a": 1} done') == '{"a": 1}'


def test_parses_and_counts_tokens():
    llm, comp = make(['{"queries": ["a", "b", "c"]}'])
    r = call(llm, Plan, {"min": 3, "max": 5})
    assert r.data.queries == ["a", "b", "c"] and r.input_tokens == 10
    assert comp.kwargs[0]["response_format"] == {"type": "json_object"}


def test_validation_error_gets_one_repair_attempt():
    llm, comp = make(['{"queries": ["a"]}', '{"queries": ["a", "b", "c", "d", "e", "f"]}'])
    r = call(llm, Plan, {"min": 3, "max": 5})
    assert r.attempts == 2 and len(r.data.queries) == 5 and r.output_tokens == 10
    assert "not valid" in comp.kwargs[1]["messages"][-1]["content"]


def test_gives_up_after_two_invalid_replies():
    llm, _ = make(["not json", '{"claims": []}'])
    with pytest.raises(LLMOutputError):
        call(llm, Draft)


def test_truncated_output_is_reported_and_limit_is_configurable():
    from research_agent.config import Settings

    class Truncating(FakeCompletions):
        async def create(self, **kwargs):
            self.kwargs.append(kwargs)
            r = resp('{"sufficient": true, "items": [{"text": "cut off')
            r.choices[0].finish_reason = "length"
            return r

    comp = Truncating([])
    llm = TokenFactoryLLM(SimpleNamespace(chat=SimpleNamespace(completions=comp)), NET, max_tokens=16384)
    with pytest.raises(LLMOutputError, match="hit the 16384-token limit"):
        call(llm, Draft)
    assert comp.kwargs[0]["max_tokens"] == 16384
    s = Settings(nebius_api_key="k")
    assert TokenFactoryLLM.from_settings(s).max_tokens == s.network.llm_max_tokens


def test_retries_transient_errors():
    llm, comp = make([openai.APIConnectionError(request=REQ), '{"sufficient": false, "claims": []}'])
    r = call(llm, Draft)
    assert r.data.sufficient is False and len(comp.kwargs) == 2


def test_non_transient_error_not_retried():
    err = openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQ), body=None)
    llm, comp = make([err, '{"sufficient": true}'])
    with pytest.raises(openai.AuthenticationError):
        call(llm, Draft)
    assert len(comp.kwargs) == 1


def test_falls_back_when_json_mode_rejected():
    err = openai.BadRequestError("response_format unsupported", response=httpx.Response(400, request=REQ), body=None)
    llm, comp = make([err, '{"sufficient": true, "claims": []}'])
    call(llm, Draft)
    assert "response_format" not in comp.kwargs[1]


def test_draft_keeps_citations_as_strings_for_the_code_check():
    # Validity is decided in code (writer.resolve_citations), so one bad ID never forces a full LLM retry.
    d = Draft.model_validate({"sufficient": True, "items": [{"text": "x", "claim_type": "fact"},
                                                            {"text": "y", "sentence_ids": ["E3-1", "[E4-2]", "N7"]}]})
    assert d.items[0].sentence_ids == [] and d.items[1].sentence_ids == ["E3-1", "E4-2", "N7"]


def test_tavily_backend_maps_responses_and_credits():
    class FakeTavily:
        async def search(self, **kw):
            self.search_kw = kw
            return {"results": [{"url": "https://fda.gov/a", "title": "A", "content": "c", "score": 1.7},
                                {"title": "no url"}], "usage": {"credits": 2}}

        async def extract(self, **kw):
            self.extract_kw = kw
            return {"results": [{"url": "https://fda.gov/a", "raw_content": "body"}],
                    "failed_results": [{"url": "https://nccn.org/b", "error": "403"}]}

    fake = FakeTavily()
    tb = TavilyBackend(fake)
    res = asyncio.run(tb.search("q", depth="advanced", max_results=10, include_domains=["fda.gov"],
                                start_date="2025-01-01"))
    assert len(res.hits) == 1 and res.hits[0].score == 1.0 and res.credits == 2
    kw = fake.search_kw
    assert kw["include_domains"] == ["fda.gov"] and kw["start_date"] == "2025-01-01"
    assert kw["search_depth"] == "advanced" and kw["max_results"] == 10 and kw["include_usage"] is True
    assert kw["auto_parameters"] is False and kw["chunks_per_source"] == 3 and kw["include_answer"] is False
    ext = asyncio.run(tb.extract(["https://fda.gov/a", "https://nccn.org/b"]))
    assert ext.pages == {"https://fda.gov/a": "body"} and ext.failed == {"https://nccn.org/b": "403"}
    assert ext.credits == 1 and ext.estimated  # no usage returned -> 1 credit per 5 successful URLs
    assert "query" not in fake.extract_kw and fake.extract_kw["extract_depth"] == "basic"
    assert fake.extract_kw["include_usage"] is True
