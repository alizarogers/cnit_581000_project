"""
Tests for the classification parser and the evaluation script, with Claude mocked.
"""

import hashlib
from pathlib import Path

import openpyxl
import pytest

from backend.analysis import parse_classification, parse_pivot
from evaluation import run_eval

TEST_SET = Path(__file__).parent.parent / "evaluation" / "Group8_Pun_Test_Set.xlsx"


@pytest.mark.parametrize("text,expected", [
    ("**Classification**: Pun", "Pun"),
    ("1. **Classification**: Non-pun\n2. **Pivot**: bank", "Non-pun"),
    ("**Classification:** Non-Pun", "Non-pun"),
    ("Classification: not a pun", "Non-pun"),
    ("**Classification**: **Pun**", "Pun"),
    ("No label here", None),
])
def test_parse_classification(text, expected):
    assert parse_classification(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("**Pivot**: interest (homographic)", "interest"),
    ("## Pivot\n**interest** (homographic)", "interest"),
    ('2. **Pivot**: "two tired", heterographic', "two tired"),
    ("- **Pivot:** hole in one (phrase-level)", "hole in one"),
    ("**Pivot**: None. No double meaning.", None),
    ("there is no pivot here", None),
    ('**Pivot**: There is no real pivot. The most ambiguous word is "chicken".', None),
    ("**Pivot**: No real pivot; closest is cast.", None),
])
def test_parse_pivot(text, expected):
    assert parse_pivot(text) == expected


class FakeLLM:
    """Calls every sentence containing 'interest' or 'cast' a pun, everything else a non-pun."""

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, history, user_message, temperature=None):
        self.calls.append((history, user_message, temperature))
        if history:
            return f"Follow-up answer to: {user_message}"
        sentence = user_message.split("\n\n---\n")[0]  # ignore the pipeline evidence block
        is_pun = any(w in sentence for w in ("interest", "cast"))
        return f"**Classification**: {'Pun' if is_pun else 'Non-pun'}\n**Pivot**: something (test)"


class FakePipeline:
    """Stage 1-2 stand-in: flags 'interest' and 'velcro' sentences."""
    threshold = 0.3

    def analyze(self, sentence):
        from backend.pipeline.stages import Candidate, PipelineResult, SenseScore
        hit = any(w in sentence for w in ("interest", "velcro"))
        best = None
        if hit:
            best = Candidate(text="interest", lemma="interest", pos="NOUN", dep="dobj", head="lost", start=0, end=8,
                             senses_total=10, senses_kept=7, double_score=0.4,
                             sense_a=SenseScore("interest.n.04", "a fixed charge", "noun.possession", 0.5, 0.9, 0.4, 0.8, "banker"),
                             sense_b=SenseScore("interest.n.01", "curiosity", "noun.cognition", 0.4, 0.9, 0.3, 0.8, "lost"))
        return PipelineResult(sentence, "Pun" if hit else "Non-pun", 0.3, best, [best] if best else [], {"parser": "x", "encoder": "y", "lexicon": "z"}, 0.1)


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(run_eval, "build_pipeline", lambda cfg, force=False: FakePipeline())


def test_run_eval_fills_copy_and_keeps_original(tmp_path, monkeypatch, fake_pipeline):
    fake = FakeLLM()
    monkeypatch.setattr(run_eval, "LLMAdapter", lambda: fake)
    before = hashlib.md5(TEST_SET.read_bytes()).hexdigest()
    out = tmp_path / "results.xlsx"

    assert run_eval.main(["--input", str(TEST_SET), "--output", str(out)]) == 0

    assert hashlib.md5(TEST_SET.read_bytes()).hexdigest() == before  # input untouched
    wb = openpyxl.load_workbook(out)
    ts = wb["Test Set"]
    assert ts["A2"].value == "P01" and ts["J2"].value == "Pun"        # banker / interest
    assert ts["A3"].value == "P02" and ts["J3"].value == "Pun"        # cast
    assert ts["J4"].value == "Non-pun"                                 # velcro, missed by fake
    assert str(ts["K2"].value).startswith("=IF(")                      # scoring formula intact
    assert all(ts.cell(r, 10).value in ("Pun", "Non-pun") for r in range(2, 32))

    dq = wb["Dialog Questions"]
    ans_col = run_eval.find_column(dq, "Chatbot answer")
    assert ans_col is not None
    assert dq.cell(2, ans_col).value == "Follow-up answer to: Why is this funny?"
    assert dq.cell(15, ans_col).value is None                          # totals rows left alone
    assert "Chatbot Output" in wb.sheetnames

    # Dialog questions are asked with the pun (plus its pipeline evidence) and its analysis as history
    follow_ups = [c for c in fake.calls if c[0]]
    assert len(follow_ups) == 12
    assert follow_ups[0][0][0]["content"].startswith("I used to be a banker, but I lost interest.")
    assert "Pipeline evidence" in follow_ups[0][0][0]["content"]
    assert "Pipeline evidence" not in follow_ups[0][1]           # the question itself is sent as typed

    out_ws = wb["Chatbot Output"]
    assert out_ws["E2"].value == "Pun" and out_ws["F2"].value == "interest"   # Stages 1-2 call and pivot
    cmp_ws = wb["System Comparison"]
    assert cmp_ws["B1"].value.startswith("Stages 1-2") and cmp_ws["C1"].value.startswith("Full system")
    assert all(c[2] == run_eval.EVAL_TEMPERATURE for c in fake.calls)
    assert out.with_suffix(".json").exists()


def test_pipeline_only_needs_no_api_key(tmp_path, monkeypatch, fake_pipeline):
    def boom():
        raise AssertionError("LLM should not be created")
    monkeypatch.setattr(run_eval, "LLMAdapter", boom)
    out = tmp_path / "p.xlsx"
    assert run_eval.main(["--input", str(TEST_SET), "--output", str(out), "--pipeline-only"]) == 0
    ts = openpyxl.load_workbook(out)["Test Set"]
    assert ts["J2"].value == "Pun" and ts["J3"].value == "Non-pun"


def test_compare_runs_claude_only_too(tmp_path, monkeypatch, fake_pipeline):
    fake = FakeLLM()
    monkeypatch.setattr(run_eval, "LLMAdapter", lambda: fake)
    out = tmp_path / "c.xlsx"
    run_eval.main(["--input", str(TEST_SET), "--output", str(out), "--only", "detection", "--compare", "--limit", "2"])
    plain = [c for c in fake.calls if "Pipeline evidence" not in c[1]]
    assert len(plain) == 2                                       # one Claude-only call per sentence
    wb = openpyxl.load_workbook(out)
    assert wb["System Comparison"]["D1"].value == "Claude only"


def test_custom_column_leaves_j_empty(tmp_path, monkeypatch, fake_pipeline):
    monkeypatch.setattr(run_eval, "LLMAdapter", lambda: FakeLLM())
    out = tmp_path / "r.xlsx"
    run_eval.main(["--input", str(TEST_SET), "--output", str(out), "--only", "detection", "--column", "M"])
    ts = openpyxl.load_workbook(out)["Test Set"]
    assert ts["J2"].value is None and ts["M2"].value == "Pun"
