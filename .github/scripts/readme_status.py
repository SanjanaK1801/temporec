"""Fill in the Status column of the interim plan table in README.md.

The README workflow runs this after every push to `interim`, i.e. after every merged PR. It
looks up each sub-task's PR (by the [TR-NN] in the title) and the "Done when" checklist of the
issue the PR closes, then rewrites the table between the status markers. Nothing else in the
README is touched.

Usage:
    python .github/scripts/readme_status.py README.md prs.json issues.json

prs.json and issues.json are the output of
    gh pr list --state all --json number,title,url,state,isDraft,mergedAt,baseRefName,body
    gh issue list --state all --json number,body
"""
import json
import re
import sys

START, END = "<!-- status:start -->", "<!-- status:end -->"
HEADER = ["| ID | Sub-task | What is delivered | Status |", "|---|---|---|---|"]


def subtask_id(title: str):
    m = re.match(r"\[(TR-\d+)\]", title or "")
    return m.group(1) if m else None


def checklist(body: str):
    """(ticked, total) boxes in an issue body."""
    ticked = len(re.findall(r"^\s*- \[[xX]\]", body or "", flags=re.M))
    open_ = len(re.findall(r"^\s*- \[ \]", body or "", flags=re.M))
    return ticked, ticked + open_


def pick_prs(prs: list) -> dict:
    """One PR per sub-task. PRs into main are ignored, and when a sub-task has several PRs
    (a follow-up, or a closed attempt) the first one that wasn't closed unmerged wins."""
    chosen = {}
    for pr in sorted(prs, key=lambda p: p["number"]):
        tr = subtask_id(pr["title"])
        if tr is None or pr["baseRefName"] == "main":
            continue
        current = chosen.get(tr)
        if current is None or (current["state"] == "CLOSED" and pr["state"] != "CLOSED"):
            chosen[tr] = pr
    return chosen


def status(pr, issues: dict) -> str:
    if pr is None:
        return "Not started"
    link = f"[#{pr['number']}]({pr['url']})"
    if pr["state"] == "MERGED":
        text = f"Merged {pr['mergedAt'][:10]} ({link})"
    elif pr["state"] == "CLOSED":
        text = f"Closed without merging ({link})"
    elif pr["isDraft"]:
        text = f"In progress ({link}, draft)"
    else:
        text = f"In review ({link})"
    m = re.search(r"Closes #(\d+)", pr.get("body") or "")
    if m and int(m.group(1)) in issues:
        ticked, total = checklist(issues[int(m.group(1))])
        if total:
            text += f", {ticked}/{total} done"
    return text


def update(readme: str, prs: list, issues: list) -> str:
    before, rest = readme.split(START, 1)
    table, after = rest.split(END, 1)
    chosen = pick_prs(prs)
    issue_bodies = {i["number"]: i["body"] for i in issues}

    rows = []
    for line in table.strip().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not re.fullmatch(r"TR-\d+", cells[0]):
            continue  # header and separator lines get rebuilt below
        rows.append(f"| {cells[0]} | {cells[1]} | {cells[2]} | {status(chosen.get(cells[0]), issue_bodies)} |")
    return before + START + "\n" + "\n".join(HEADER + rows) + "\n" + END + after


def main() -> None:
    readme_path, prs_path, issues_path = sys.argv[1:4]
    with open(readme_path) as f:
        readme = f.read()
    with open(prs_path) as f:
        prs = json.load(f)
    with open(issues_path) as f:
        issues = json.load(f)
    with open(readme_path, "w") as f:
        f.write(update(readme, prs, issues))


if __name__ == "__main__":
    main()
