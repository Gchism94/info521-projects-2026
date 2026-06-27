# Checkpoint 3 — Identify the Posterior for a Conjugate Pair

**Gates:** Project 1.3 (Bayesian)
**Format:** in-class, closed-book, ~15 min. Graded Satisfactory / Not-Yet.
**Retakeable.**

This checkpoint tests the study-guide outcome *"given a conjugate
prior–likelihood pair, identify the form of the posterior."* It uses
Beta–Binomial as the clean, time-boxed instance. The project applies the **same
principle** to the Gaussian–Gaussian pair — make that connection explicit when
you hand it back.

---

## Prompt (Version A)

A coin has unknown heads probability $r$. You place a prior
$r \sim \text{Beta}(\alpha, \beta)$ and observe $y$ heads in $N$ tosses
(Binomial likelihood). Identify the posterior distribution of $r$ — name the
family and give its parameters. State in one sentence what makes this pair
*conjugate*.

## Prompt (Version B — retake)

You place a Gaussian prior $w \sim \mathcal{N}(0, \tau^2)$ on a single weight in a
Gaussian-noise linear model. Without full derivation, name the family of the
posterior $p(w \mid \text{data})$ and explain why conjugacy guarantees it stays
in that family.

---

## Satisfactory criteria

- Names the correct posterior family.
- (A) Gives correct updated parameters.
- States conjugacy = prior and posterior share a family (posterior stays
  analytically tractable).

## Answer key

**A:** Posterior is $\text{Beta}(\alpha + y,\ \beta + N - y)$. Conjugate because
the Beta prior and Binomial likelihood combine to yield another Beta.

**B:** Posterior is **Gaussian**. A Gaussian prior with a Gaussian-noise
likelihood yields a Gaussian posterior, so it remains closed-form — the exact
property Project 1.3 relies on and Project 2 deliberately breaks.
