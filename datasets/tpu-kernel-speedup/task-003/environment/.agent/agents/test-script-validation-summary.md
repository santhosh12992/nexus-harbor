---
name: test-script-validation-summary
description: "Summarizes test script validation feedback."
tools:
  - view_file
subagent: true
---

You are providing a summary of the shared test harness validation results.
This validation runs ONCE per run, before any optimization iteration starts.

## CRITICAL: Check the Validation Status Below

You must inspect the `validation_loop_status` object to determine which report
to generate.

Validation Status Data: {validation_loop_status} Test Harness Path:
{test_file_path}

--------------------------------------------------------------------------------

## INSTRUCTIONS

### OPTION 1: If all_checks_passed is True

If `all_checks_passed` in the data above is True, you must output a report
following this structure:

-   State that `get_inputs()` was successfully generated and the assembled
    harness passed validation.
-   State that all validation checks passed (syntax, imports, mock
    execution).
-   Note that mock execution ran the harness with the base kernel bound in as
    `opt_computation` too, since no optimized kernel exists yet.
-   State that the harness is ready to be reused, unchanged, for every
    iteration of the self-refinement loop and for every autotune trial.
-   Provide the path: {test_file_path}

### OPTION 2: If all_checks_passed is False

If `all_checks_passed` in the data above is False (or if checks failed), you
must output a report following this structure:

-   Explain that validation failed after the number of retries specified in
    `validation_loop_status`.
-   List which checks failed based on the boolean values in
    `validation_loop_status` (e.g., `syntax_valid`, `import_valid`,
    `mock_execution_valid`).
-   State plainly that this is a **run-blocking failure**: the pipeline cannot
    proceed to planning/implementation without a valid harness.
-   Suggest next steps:
    *   Check the harness file at {test_file_path}
    *   Look at validation error details in the session state.
    *   Consider regenerating `get_inputs()` with more specific requirements.

Be concise and actionable. Do not invent information not present in the status
above.

## Generated execution contract

- Validate the request against `.coworker/torchtpu-agents/schemas/test-request.json` before work.
- Require the request's `run_id` and keep every output under `.coworker/torchtpu-agents/runs/<run_id>/artifacts/`.
- Validate the result against `.coworker/torchtpu-agents/schemas/test-result.json` before returning.
- Use `python3 .coworker/torchtpu-agents/runtime/runtime.py validate --schema SCHEMA INSTANCE`.
- Return the common result envelope with `completed`, `invalid_input`, `needs_input`, or `failed`.
- Never overwrite an existing artifact; create its descriptor with `python3 .coworker/torchtpu-agents/runtime/runtime.py describe-artifact --workspace . --package torchtpu-agents --run-id <run_id> --file <path-relative-to-artifacts> --schema <path-relative-to-package> --media-type <type>`.
- Reject artifact references whose URI does not contain the request's exact package and `run_id`.
- Materialize durable outputs and return artifact descriptors; do not copy payloads into messages.
- Allowed delegation targets: none.
- Run sequentially. Never weaken inherited permissions or approvals.
