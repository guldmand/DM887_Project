# Plan: Results, Discussion, and Conclusion

## 0. Purpose

This file defines how to write results, discussion, and conclusion for the DM887 midway report and later final report.

---

## 1. Results section

The results section should present observed outputs without overinterpreting them.

For midway, include either:

1. learning curves from completed baseline runs, or
2. a run-status table if full curves are not ready, or
3. both.

### Results should include

- algorithms,
- environments,
- seeds completed,
- training steps,
- evaluation returns,
- figure or table.

### Results should avoid

- unsupported claims,
- long explanations of algorithm theory,
- subjective interpretation,
- repeating every number from the figure.

---

## 2. Example results wording

If results are partial:

```text
Figure X shows the available interim baseline learning curves. The runs are intended to validate the ObjectRL evaluation pipeline and should be interpreted as preliminary because the full five-seed training budget has not yet been completed.
```

If results are complete:

```text
Figure X shows the five-seed baseline learning curves for PPO, SAC, and TD3. Each curve reports the mean undiscounted evaluation return as a function of training steps, with shaded regions indicating variation across seeds.
```

---

## 3. Discussion section

The discussion interprets the results in relation to the project goal.

Questions to answer:

1. Did the baselines run as expected?
2. Which algorithm appears strongest and why might that be?
3. Are results stable across seeds?
4. Are there environment-specific difficulties?
5. What are the limitations of the current evidence?
6. What must be improved for the final report?

---

## 4. Safe interpretation language

Use:

```text
The preliminary result suggests...
```

```text
This may indicate...
```

```text
A full conclusion requires the complete five-seed experiment matrix.
```

Avoid:

```text
This proves that SAC is better.
```

unless the evidence and statistical support are strong.

---

## 5. Conclusion section

The conclusion should be short and clear.

For midway, include:

1. what was established,
2. what remains,
3. why it matters for the final project.

Suggested midway conclusion:

```text
This interim report established the related work, notation, and baseline protocol for the GRPO-control project. PPO, SAC, and TD3 were selected as the required baselines and integrated into an ObjectRL-based experimental workflow. The next step is to complete the full baseline matrix, implement the GRPO-control variant, and compare its performance and theoretical properties against the baseline methods.
```

---

## 6. Common mistakes

Results mistakes:

- unclear figure captions,
- no axis labels,
- no seed information,
- showing raw logs instead of processed results.

Discussion mistakes:

- overclaiming,
- introducing new results,
- ignoring failed runs,
- not connecting to the research question.

Conclusion mistakes:

- vague take-home message,
- new data introduced,
- limitations omitted,
- future work missing.

---

## 7. Final project extension

For the final report, the discussion must also compare:

- GRPO-control vs PPO,
- GRPO-control vs SAC,
- GRPO-control vs TD3,
- environment-specific differences,
- whether GRPO achieved the stated hypothesis,
- whether the theory assumptions seem plausible in practice.
