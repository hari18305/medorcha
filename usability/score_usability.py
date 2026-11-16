"""Score the usability study: SUS per participant and version, and comprehension results per version.

Usage (from the project folder):
    python usability/score_usability.py                      # reads usability/data/responses.csv
    python usability/score_usability.py path/to/file.csv
    python usability/score_usability.py --selftest           # checks the arithmetic, reads no data

Input: one row per participant per version (see usability/results_template.csv).
  sus1..sus10   System Usability Scale answers, 1 to 5
  x11..x14      our own statements, 1 to 5 (reported as means, not part of the SUS)
  q1..q6        comprehension questions, 1 = correct, 0 = incorrect
  q7_trust      willingness to act on the result, 1 to 5 (reported as a mean)
Blank cells are skipped. Standard library only.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "responses.csv"
QUESTIONS = {
    "q1": "What the number means",
    "q2": "What the missing points are",
    "q3": "High score can still be wrong",
    "q4": "Low score can still be right",
    "q5": "Why the case needed review",
    "q6": "What the image statement means",
}
OWN_ITEMS = {
    "x11": "Understood why the system reached its result",
    "x12": "Understood how much to trust the result",
    "x13": "Wording clear without technical explanation",
    "x14": "Could find the evidence behind a diagnosis",
}


def sus_score(answers: list[int]) -> float:
    """SUS (Brooke, 1996): odd items contribute answer-1, even items 5-answer; the sum is multiplied by 2.5."""
    if len(answers) != 10 or any(a < 1 or a > 5 for a in answers):
        raise ValueError("SUS needs ten answers between 1 and 5")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))  # index 0 is item 1 (odd)
    return total * 2.5


def _num(value: str):
    value = (value or "").strip()
    return float(value) if value else None


def _mean(values: list[float]):
    return statistics.mean(values) if values else None


def _sd(values: list[float]):
    return statistics.stdev(values) if len(values) > 1 else None


def _fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def analyse(rows: list[dict]) -> str:
    by_version: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_version[(row.get("version") or "?").strip().upper()].append(row)

    out: list[str] = []
    out.append("Participants per version: " + ", ".join(f"{v} = {len(r)}" for v, r in sorted(by_version.items())))
    out.append("")

    out.append("SUS (0-100; higher is better)")
    out.append(f"{'Version':<9}{'n':>3}{'mean':>8}{'SD':>7}{'min':>7}{'max':>7}   individual scores")
    for version, version_rows in sorted(by_version.items()):
        scores = []
        for row in version_rows:
            answers = [_num(row.get(f"sus{i}")) for i in range(1, 11)]
            if all(a is not None for a in answers):
                scores.append(sus_score([int(a) for a in answers]))
        out.append(
            f"{version:<9}{len(scores):>3}{_fmt(_mean(scores)):>8}{_fmt(_sd(scores)):>7}"
            f"{_fmt(min(scores) if scores else None):>7}{_fmt(max(scores) if scores else None):>7}   "
            + ", ".join(f"{s:.1f}" for s in scores)
        )
    out.append("")

    out.append("Comprehension: percentage answering correctly (correct / answered)")
    versions = sorted(by_version)
    out.append(f"{'Question':<40}" + "".join(f"{v:>16}" for v in versions))
    for key, label in QUESTIONS.items():
        cells = []
        for version in versions:
            answers = [_num(r.get(key)) for r in by_version[version]]
            answers = [a for a in answers if a is not None]
            cells.append(f"{100 * sum(answers) / len(answers):.0f}% ({int(sum(answers))}/{len(answers)})" if answers else "n/a")
        out.append(f"{key + '  ' + label:<40}" + "".join(f"{c:>16}" for c in cells))
    out.append("")

    out.append("Ratings, mean on a 1-5 scale")
    out.append(f"{'Item':<52}" + "".join(f"{v:>8}" for v in versions))
    items = {"q7_trust": "Willingness to act on the result", **OWN_ITEMS}
    for key, label in items.items():
        cells = []
        for version in versions:
            values = [_num(r.get(key)) for r in by_version[version]]
            cells.append(_fmt(_mean([v for v in values if v is not None]), 2))
        out.append(f"{label:<52}" + "".join(f"{c:>8}" for c in cells))
    out.append("")
    out.append("Small sample: describe these as formative findings, not as a statistical comparison.")
    return "\n".join(out)


def selftest() -> None:
    assert sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]) == 100.0  # best possible
    assert sus_score([1, 5, 1, 5, 1, 5, 1, 5, 1, 5]) == 0.0    # worst possible
    assert sus_score([3] * 10) == 50.0
    assert sus_score([4, 2, 4, 2, 4, 2, 4, 2, 4, 2]) == 75.0
    try:
        sus_score([6] + [3] * 9)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range answer must be rejected")
    sample = [
        {"version": "A", **{f"sus{i}": "3" for i in range(1, 11)}, "q1": "0", "q2": "1", "q7_trust": "2"},
        {"version": "A", **{f"sus{i}": "3" for i in range(1, 11)}, "q1": "1", "q2": "1", "q7_trust": "4"},
        {"version": "B", **{f"sus{i}": ("5" if i % 2 else "1") for i in range(1, 11)}, "q1": "1", "q2": "1", "q7_trust": "5"},
    ]
    report = analyse(sample)
    assert "A = 2" in report and "B = 1" in report
    assert "50.0" in report and "100.0" in report
    assert "50% (1/2)" in report and "100% (1/1)" in report
    print("self-test passed")


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        selftest()
        return 0
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"No data file at {path}. Copy usability/results_template.csv to usability/data/responses.csv and fill it in.")
        return 1
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if any((v or "").strip() for v in row.values())]
    if not rows:
        print("The file has no data rows yet.")
        return 1
    print(analyse(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
