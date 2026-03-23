# Coding Guidelines — .ai_agents/coding_guidelines.md


Two modes of working:
- feature:
- fix:
You *must* ask the operator which mode they would like to work in if not specified
Feature mode consists of planning and implementation stage
1. Planning stage: You are *not* allowed to write code. Instead plan what needs to be done addressing the following. place: each plan (including markdown and diagrams) in feature_plans/... where ... is of the form 001_feature_summary_timestamp
    - Detailed plan of the changes you propose to make
    - Files you will touch / create to do this
    - Classes and / or Functions you will create
    - Class Diagrams and Sequence Diagrams showing what you are creating and how it interacts with existing logic (only if it interacts with existing logic)
    - Constraints
    - Goals 
    - Key tests that you will use to verify this
    - Stages how you will implementation (you can have 1 stage if the feature can accept this) Each stage must be self sufficient, i.e. you add tests, code to pass tests, run validation pipelines and confirm success (and ideally commit) before moving on to the next stage. Generally if you anticipate making more than 500 lines of code changes in a stage it should be broken down into smaller changes. By default each stage will be a commit unless i ask you to work without commiting. 
Come to me when done. I will review and if necessary ask you to make changes. You cannot move to the implementation stage until I have approved. 
2. Implementation Stage:
    When the plan is approved you may start the implementation. Work on one stage at a time.
    For each stage:
        - Write tests first that will fail
        - Write code to implement the stage and check the tests pass
        - Refine as necessary
        - Run validation checls, e.g. type checker, linter and formatter
        - Finish a stage when you are confident the feature works, tests pass, and validation checks pass
    Keep a section of the plan called progress that you update as work on each stage to allow me to track how you are doing. 
    When you are done with all stages, run over all changes again and check it is suitable according to the coding guidelines

Fix mode consists of direct implementation. Just confirm to me in the chat you understand what to do then make the fix (this is generally for small areas of work.)



General
- Keep changes small and atomic. 
- Update `.ai_agents/understanding.md` and `.ai_agents/task.md` for every task change.
- Favor simplicity over unnecessary abstraction; keep implementations as simple as possible while meeting requirements.
- Do not add `# noqa` suppressions without explicit approval.
- Fail fast: surface errors clearly rather than suppressing or masking them.

Python:
- Always use type hints 
- Do not suppress type checking
- Avoid writing lots of standalone functions, group common functionality into classes, prefer static methods in classes than single functions. 
- Opt for pydantic models for parsing any data
- pytest framework

Testing
- Each function should have a unit test
- Each feature should have integration tests

Documentation
- Update README with run instructions, ports, and env variables.
- Keep `.ai_agents/understanding.md` authoritative for architecture.

Agent / developer policy

- Required tooling for all development features:
  - Formatter & linter: use `ruff` for both formatting and linting. Use `ruff format` (or `ruff --fix`) to apply formatting changes and `ruff check` to run lint checks.
  - Type checker: use `PyreFly` for static typing checks (fail fast, no suppression).
  - Tests: use `pytest` following Test-Driven Development (TDD). Every feature MUST have tests written first and all tests must pass before marking the feature complete.
  - CI / PR requirements: every change that implements a feature must run (in this order): `ruff format` (or fixes applied), `ruff check`, `PyreFly` type-check, and `pytest`.

Development rules:

  - Follow these coding rules for style, architecture, and testing expectations.
  - Add or update unit tests with each feature following TDD. Tests should be deterministic and fast.
  - Add typing (type hints) for public functions and data models. Run `PyreFly` and fix errors before merging.
  - Use dependency injection and small modules to keep code testable and follow SOLID principles as described in the project docs.
  - If i make a correction to you you must document it in `.ai_agents/learnings.md` in the structure:
    - Issue:
    - What misunderstanding happened:
    - Why it was wrong:
    - What to do next time:
    You must read this file before working on any feature. 

Operational rule for automated agents:

  - If you are an automated agent modifying the repo, you MUST update `.ai_agents/task.md` with a timestamped entry describing the change and set the corresponding todo status.
