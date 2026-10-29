import importlib.util
from pathlib import Path

script = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "readme_status.py"
spec = importlib.util.spec_from_file_location("readme_status", script)
readme_status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readme_status)

README = """intro
<!-- status:start -->
| ID | Sub-task | What is delivered |
|---|---|---|
| TR-01 | Scaffold | The repo |
| TR-02 | Jobs | Slurm scripts |
| TR-03 | Pipeline | Clean data |
<!-- status:end -->
outro
"""


def pr(number, title, state, base="interim", draft=False, merged=None, closes=None):
    return {"number": number, "title": title, "url": f"https://example/pull/{number}", "state": state,
            "isDraft": draft, "mergedAt": merged, "baseRefName": base,
            "body": f"Closes #{closes}\n" if closes else ""}


PRS = [
    pr(12, "[TR-01] Scaffold", "MERGED", merged="2026-10-29T10:00:00Z", closes=2),
    pr(13, "[TR-02] Jobs", "OPEN", draft=True, closes=3),
    pr(21, "[TR-01] Add project README to main", "MERGED", base="main", merged="2026-10-27T00:11:55Z"),
]
ISSUES = [{"number": 2, "body": "- [x] one\n- [x] two\n- [ ] three"}, {"number": 3, "body": "no checklist"}]


def test_status_column_is_filled_in():
    lines = readme_status.update(README, PRS, ISSUES).splitlines()
    assert lines[2] == "| ID | Sub-task | What is delivered | Status |"
    assert lines[4] == "| TR-01 | Scaffold | The repo | Merged 2026-10-29 ([#12](https://example/pull/12)), 2/3 done |"
    assert lines[5] == "| TR-02 | Jobs | Slurm scripts | In progress ([#13](https://example/pull/13), draft) |"
    assert lines[6] == "| TR-03 | Pipeline | Clean data | Not started |"
    assert lines[0] == "intro" and lines[-1] == "outro"


def test_running_twice_changes_nothing():
    once = readme_status.update(README, PRS, ISSUES)
    assert readme_status.update(once, PRS, ISSUES) == once


def test_a_closed_attempt_gives_way_to_a_newer_pr():
    prs = [pr(13, "[TR-02] Jobs", "CLOSED"), pr(30, "[TR-02] Jobs, second try", "OPEN")]
    assert readme_status.pick_prs(prs)["TR-02"]["number"] == 30
