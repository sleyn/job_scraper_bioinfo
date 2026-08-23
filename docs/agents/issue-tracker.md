# Issue tracker: Local Markdown

This repo has no git remote, and work is tracked by a single developer. Issues and specs
live as markdown files in `.scratch/`. `TODO.md` at the repo root remains the human-facing
roadmap of deferred work; `.scratch/` is where a specific piece of work gets broken down
before it is built.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue
number directly.

## Not configured

- **Triage.** The `triage` skill is not installed here, so there is no `Status:` line
  convention and no `docs/agents/triage-labels.md`. Don't invent one.
- **Wayfinding.** The `wayfinder` skill is not installed, so there are no `map.md` efforts.
- **PRs as a request surface.** No remote, so no PR queue.
