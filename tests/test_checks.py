"""Deterministic checks and identifier detection."""

import pytest

from research_agent.engine.checks import (POPULATION_NOT_STATED, check_design, check_numbers, check_population,
                                          numbers_in)
from research_agent.evidence import near_duplicate
from research_agent.privacy import find_identifiers
from research_agent.sources import classify, tier_of

EXCERPT = "Median PFS was 23.7 months vs 16.6 months (HR 0.70; 95% CI, 0.58-0.85; p<0.001) in the phase III trial."


def test_numbers_ignore_names_and_normalize():
    assert numbers_in("HER2 KRAS G12C CDK4/6 PD-1 phase 3 exon 19") == set()
    assert numbers_in("HR 0.70 and 1,200 patients") == {"0.7", "1200"}


def test_number_check_passes_and_fails():
    cit = [{"evidence_id": 1, "excerpt": EXCERPT, "source_sentence": EXCERPT}]
    assert check_numbers("PFS 23.7 vs 16.6 months (HR 0.7)", cit) == (True, [])
    ok, problems = check_numbers("PFS 24.1 months (HR 0.70)", cit)
    assert not ok and "24.1" in problems[-1]
    fake = [{"evidence_id": 1, "excerpt": EXCERPT, "source_sentence": "HR 0.68 overall."}]
    ok, problems = check_numbers("HR 0.68", fake)
    assert not ok and "not in the excerpt" in problems[0]
    assert check_numbers("No numbers here", []) == (True, [])


def test_design_must_be_stated():
    assert check_design("phase 3 randomized trial", ["A phase III randomised trial"]) == "phase 3 randomized trial"
    assert check_design("phase 3 randomized trial", ["A single-arm phase 2 study"]) == "not stated"
    assert check_design("regulatory decision", ["FDA approved"]) == "regulatory decision"
    assert check_design("", ["x"]) == "not stated"


def test_population_requires_verified_note():
    notes = [{"evidence_ids": [1], "details": {"population": "adults 18-75 in Europe",
                                               "population_sentence": "We enrolled adults 18-75 in Europe."}}]
    assert check_population("adults 18-75 in Europe", notes, {1}) == "adults 18-75 in Europe"
    assert check_population("adults 18-75 in Europe", notes, {2}) == POPULATION_NOT_STATED  # not a cited source
    assert check_population("children in Japan", notes, {1}) == POPULATION_NOT_STATED
    assert check_population(None, notes, {1}) == POPULATION_NOT_STATED


def test_near_duplicate_queries():
    assert near_duplicate("FDA approval drugx EGFR NSCLC", ["drugx egfr nsclc fda approval"])
    assert not near_duplicate("drugx overall survival", ["drugx egfr nsclc fda approval"])


def test_tiers_and_source_types():
    assert tier_of("meetings.asco.org") == 1 and tier_of("www.nejm.org") == 2 and tier_of("onclive.com") == 3
    assert tier_of("example.com") is None
    assert classify("https://www.fda.gov/news/x", "fda.gov") == "regulatory_action"
    assert classify("https://clinicaltrials.gov/study/NCT01", "clinicaltrials.gov") == "registry_entry"
    assert classify("https://ascopubs.org/doi/10.1200/JCO.2025.43.16_suppl.8500", "ascopubs.org") == "conference_abstract"
    assert classify("https://www.nejm.org/doi/full/10.1056/x", "nejm.org") == "peer_reviewed_article"
    assert classify("https://www.prnewswire.com/x", "prnewswire.com") == "press_release"
    assert classify("https://www.esmo.org/guidelines/lung", "esmo.org") == "guideline"
    assert classify("https://www.cancer.gov/about-cancer", "cancer.gov") is None  # ambiguous -> worker decides


@pytest.mark.parametrize("text", [
    "What changed for EGFR NSCLC since 2025-06-01?",
    "Updates to NCT04294810 and KEYNOTE-189 since June 2025?",
    "Patient Reported Outcomes for HER2-low breast cancer since 2024",
    "Dose 1000 mg in stage IV CRC: any change since 01/06/2025?",
])
def test_clinical_questions_are_not_flagged(text):
    assert find_identifiers(text) == []


@pytest.mark.parametrize("text,label", [
    ("MRN: A12345 EGFR NSCLC", "medical record number"),
    ("record 98765432 metastatic breast", "MRN-like number"),
    ("DOB 1961-03-14 with melanoma", "date of birth"),
    ("born March 14, 1961, AML", "date of birth"),
    ("my patient Jane Doe has CLL", "patient name"),
    ("Mr. Okafor has prostate cancer", "patient name"),
    ("SSN 123-45-6789", "social security number"),
])
def test_identifiers_are_flagged(text, label):
    assert label in find_identifiers(text)
