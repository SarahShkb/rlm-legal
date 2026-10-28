# Qwen3-235B-A22B-2507 root pilot (stopped)

Setup: root qwen/qwen3-235b-a22b-2507 (OpenRouter parasail/fp8), 32K root window enforced
client-side, Qwen3-235B as its own sub-LLM, no forced delegation, 12 iterations,
planned contracts 0 19 38 57 76 95 114 133.

Stopped by hand on 2026-10-28 after 3 contracts; only contract 0 has a record.

| Contract | Outcome |
|---|---|
| 0 | finished: 2 sub-LLM calls, answer not readable (wrong) |
| 19 | hung: REPL code ran at 100% CPU for 39 min; the in-process watchdog could not interrupt it; process killed, no record |
| 38 | stopped after ~20 min in iteration 2: one search block took 662 s (`context.upper().find(...)` inside a loop over every character of a 300K-character contract); no record |

Why stopped: the pilot's pass criteria (sub-LLM used in at least 4 of 8 runs and at least
5 of 8 correct) could no longer be met, and each remaining contract could take up to the
30-minute watchdog limit. Qwen3-235B writes REPL code too slow for MAUD-length contracts.
