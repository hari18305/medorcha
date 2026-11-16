# System Usability Scale and feedback form (participant copy)

Please answer after using **this version** of the screen. Circle one number for each statement:
**1 = strongly disagree, 5 = strongly agree.** Answer from your first impression; there is no wrong answer.

Participant: ______   Version used: A / B   (facilitator fills in)

| # | Statement | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | I think that I would like to use this system frequently. | 1 | 2 | 3 | 4 | 5 |
| 2 | I found the system unnecessarily complex. | 1 | 2 | 3 | 4 | 5 |
| 3 | I thought the system was easy to use. | 1 | 2 | 3 | 4 | 5 |
| 4 | I think that I would need the support of a technical person to be able to use this system. | 1 | 2 | 3 | 4 | 5 |
| 5 | I found the various functions in this system were well integrated. | 1 | 2 | 3 | 4 | 5 |
| 6 | I thought there was too much inconsistency in this system. | 1 | 2 | 3 | 4 | 5 |
| 7 | I would imagine that most people would learn to use this system very quickly. | 1 | 2 | 3 | 4 | 5 |
| 8 | I found the system very cumbersome to use. | 1 | 2 | 3 | 4 | 5 |
| 9 | I felt very confident using the system. | 1 | 2 | 3 | 4 | 5 |
| 10 | I needed to learn a lot of things before I could get going with this system. | 1 | 2 | 3 | 4 | 5 |

## A few more questions (same scale, 1 = strongly disagree, 5 = strongly agree)

| # | Statement | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 11 | I understood why the system reached its result. | 1 | 2 | 3 | 4 | 5 |
| 12 | I understood how much to trust the result. | 1 | 2 | 3 | 4 | 5 |
| 13 | The wording was clear to me, without technical explanation. | 1 | 2 | 3 | 4 | 5 |
| 14 | I could find the evidence behind a diagnosis when I wanted it. | 1 | 2 | 3 | 4 | 5 |

## In your own words
- Anything confusing or unclear? ______________________________________________
- Anything missing that you wanted to know? ______________________________________
- Any word or number you did not understand? ______________________________________

---

## Scoring (facilitator)
The ten SUS statements (1 to 10) are from J. Brooke, "SUS: a quick and dirty usability scale," 1996, and are free to use with that
attribution. For each participant and version:
- For odd-numbered statements (1, 3, 5, 7, 9): contribution = answer − 1.
- For even-numbered statements (2, 4, 6, 8, 10): contribution = 5 − answer.
- SUS = (sum of the ten contributions) × 2.5, giving 0 to 100.

A score of about 68 is often quoted as an average benchmark; with five people treat any comparison as indicative, not proven.
Statements 11 to 14 are our own and are reported as simple means, not as part of the SUS.
`score_usability.py` does this arithmetic from the CSV.
