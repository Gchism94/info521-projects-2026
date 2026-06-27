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

Project 2 checkpoints are defined alongside its scaffold (candidates:
Newton–Raphson update; precision/recall/F1; k-means objective and choosing K).

## Porting to Typst

These are markdown for portability. If you want them in the same Typst pipeline
as the exams, they map directly onto your `common.typ` / `#answer` / `#points`
infrastructure — the prompt becomes the question body and the answer key becomes
the `#answer` block.
