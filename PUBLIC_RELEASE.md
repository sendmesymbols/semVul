# Public release policy

This repository is the research workspace, not the publication artifact. Do not
publish the workspace directly. Build an allowlisted artifact with
`scripts/build_public_release.ps1`.

## Profiles

- `Code`: source code and launchers only. Use this for the smallest public code
  release. Dataset and generated-artifact locations are represented by empty
  directories with explanatory README placeholder files.
- `Reviewer`: the code profile plus the four audited CSV splits, the four
  canonical `ACTIVE` explanation files, the six final L1--L3 result-cache
  families, and generated figures. Use this only if the venue permits those
  artifacts and their licenses allow redistribution.

The builder copies files into a new directory. It does not remove or modify the
research workspace.

```powershell
# Minimal public source package
.\scripts\build_public_release.ps1 -Profile Code

# Reproduction artifact for reviewers
.\scripts\build_public_release.ps1 -Profile Reviewer
```

## Included source

- The `src/` data-loading, quality-feature, configuration, and RQ-analysis
  modules reached by the documented final entry points.
- `experiments/fusevul_ladder/{data,model,train}.py`: the trainer used by the
  final ladder.
- The four explanation-pipeline source files under
  `experiments/explanation/`.
- `experiments/expl_enrich/reproduce_real.py`.
- The six final per-rung launchers for PowerShell and Bash, the two explanation
  launchers, matching cache-completion helpers, `requirements.txt`, `.gitignore`,
  and the public README.

## Deliberately excluded

- Internal working material: `Critique.md`, `Observations.md`, `bug.txt`, issue
  screenshots, design notes, takeover notes, and exploratory reports.
- Duplicate or third-party snapshots: `FuSEVul-main/`, `FuSEVul-main.zip`,
  `Original FuseVul Dataset/`, `Ubuntu Run/`, `REVEAL.py`, and `reveal_cl.py`.
- Scratch and obsolete experiments: enrichment trials, pilots, probes, smoke
  outputs, Claude-generation work areas, and non-final run families.
- Legacy Python pipelines not reached by the documented entry points, including
  the old frozen-embedding/LoRA stack under `src/`, ladder convenience runners,
  and repository-internal tests.
- Generated intermediates: `experiments/cache/`, model downloads, checkpoints,
  virtual environments, IDE metadata, and the many non-canonical explanation
  variants.
- Convenience scripts that do not define the documented final pipeline.

Exclusion from the artifact does not imply that a file is safe to delete from
the private research archive.

## Checks required before publication

1. Add a project license. There is currently no `LICENSE` file. Do not select a
   license for third-party code or data until their upstream terms have been
   checked.
2. If review is double-blind, keep the builder's anonymous default. It strips
   the acknowledgments section from the copied README and fails if known
   identifying names remain in text-like release files. Pass `-KeepIdentities`
   only for a non-anonymous public release. Keep identifying metadata out of
   archive names, commit authors, document properties, and generated files.
3. Confirm redistribution rights for Devign, ReVeal, generated explanations,
   model outputs, figures, and any adapted FuSEVul code before using the
   `Reviewer` profile.
4. Run secret and personal-data scanning against both the current Git history and
   the built artifact. Removing a secret from the latest commit does not remove
   it from Git history.
5. Build from a clean, dedicated public branch or a fresh repository. Do not push
   the private repository history if it contains excluded material.
6. Run the smoke/reproduction commands from the built artifact in a clean
   environment before uploading it.
