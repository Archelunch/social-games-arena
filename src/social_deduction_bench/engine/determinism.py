"""Determinism harness: the test utility that enforces benchmark invariant #4.

Invariant #4 says everything stochastic derives from the engine seed, so the
same seed produces the same game — replayable, byte-for-byte. A harness that
*asserts* this is what turns the invariant from an aspiration into a gate: a
seeded producer is run twice and the two transcripts are compared.

The comparator deliberately does not use `EventStream ==`. `EventLog` defines
no `__eq__`, so a frozen `EventStream`'s generated equality compares its `log`
field by identity — two streams with identical content are never `==`.
`assert_streams_identical` instead compares the `header` (a frozen, structural
dataclass) and the `log.events` tuples (`tuple[Event, ...]` of frozen `Event`s,
also structural) explicitly, and reports the *first* divergence with its `seq`
so a determinism break is debuggable rather than just "streams differ".

All checks `raise AssertionError` explicitly rather than using bare `assert`,
so the harness keeps working under `python -O` — fail-loud, matching the rest
of `engine/`.
"""

from collections.abc import Callable

from social_deduction_bench.engine.events import EventStream


def assert_streams_identical(actual: EventStream, expected: EventStream) -> None:
    """Raise `AssertionError` on the first divergence between two transcripts.

    Compares structurally — never via `EventStream ==`, which is identity-only
    on the `log` field (see the module docstring). Checks run in a fixed order
    so the reported reason is deterministic:

    1. headers (seed and roster — what replay reconstructs the game from),
    2. event counts (a truncated or padded log is a determinism break),
    3. events pairwise; the message names the first divergent event's `seq`,
    4. JSONL serialization, as a belt-and-suspenders byte-level check.

    Reports only the *first* divergent position so a break is easy to debug.
    """
    if actual.header != expected.header:
        raise AssertionError(f"stream headers differ:\n  actual:   {actual.header}\n  expected: {expected.header}")

    actual_events = actual.log.events
    expected_events = expected.log.events
    if len(actual_events) != len(expected_events):
        raise AssertionError(
            f"event-count mismatch: actual has {len(actual_events)}, expected has {len(expected_events)}"
        )

    for actual_event, expected_event in zip(actual_events, expected_events, strict=True):
        if actual_event != expected_event:
            raise AssertionError(
                f"events diverge at seq {actual_event.seq}:\n  actual:   {actual_event}\n  expected: {expected_event}"
            )

    actual_lines = list(actual.to_jsonl_lines())
    expected_lines = list(expected.to_jsonl_lines())
    if actual_lines != expected_lines:
        for line_number, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines, strict=False)):
            if actual_line != expected_line:
                raise AssertionError(
                    f"JSONL serialization diverges at line {line_number}:\n"
                    f"  actual:   {actual_line}\n"
                    f"  expected: {expected_line}"
                )
        raise AssertionError(
            f"JSONL line-count mismatch: actual has {len(actual_lines)}, expected has {len(expected_lines)}"
        )


def assert_deterministic(produce: Callable[[int], EventStream], seed: int) -> None:
    """Run `produce(seed)` twice and assert the two transcripts are identical.

    This is the harness's core promise (invariant #4): a producer driven only
    by the engine seed yields the same event stream on every run. A producer
    that ignores its seed — or reaches for wall-clock state — diverges on the
    second run and is caught by `assert_streams_identical`.
    """
    first = produce(seed)
    second = produce(seed)
    assert_streams_identical(first, second)
