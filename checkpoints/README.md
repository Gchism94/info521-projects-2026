# Checkpoints

Short, closed-book, in-class derivation slips that gate the project milestones.
They carry the unaided-derivation and recall outcomes the former exams tested,
without a high-stakes timed event.

- ~15–20 minutes, **one derivation** each.
- Graded **pass / retake** (Satisfactory / Not-Yet). Each file ships an alternate
  **Version B** for retakes.
- A milestone cannot be marked Satisfactory until its gating checkpoint passes.
- Answer keys are included (end-result expressions) for fast, consistent grading
  and so you can self-check, per the course-improvement note.

| Checkpoint | Gates | Topic |
|---|---|---|
| 1 | Project 1.1 | Normal equations |
| 2 | Project 1.2 | MLE for the weights |
| 3 | Project 1.3 | Identify the posterior of a conjugate pair |
| 4 | Project 2.1 / Gate A | Newton–Raphson update for the MAP (Bayesian logistic regression) |
| 5 | Project 2.3 / Gate B | k-means objective & choosing K; PCA (retake) |

Each Project 2 gate carries one checkpoint: **Checkpoint 4** gates Gate A (the
supervised pair 2.1 + 2.2), **Checkpoint 5** gates Gate B (the unsupervised pair
2.3 + 2.4). The 2.5 capstone is gated by a structured peer review, not a
checkpoint (see `../instructor/peer-review.md`).

## Porting to Typst

These are markdown for portability. If you want them in the same Typst pipeline
as the exams, they map directly onto your `common.typ` / `#answer` / `#points`
infrastructure — the prompt becomes the question body and the answer key becomes
the `#answer` block.
