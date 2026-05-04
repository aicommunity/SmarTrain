# Breaking Change Notes for Canonical Cutover

## Potential Breaks

- Legacy field names replaced by canonical names in internal read path.
- Some source-specific optional fields may be dropped.
- Analyze/read logic may require canonical manifests for new runs.

## Mitigation

- Provide migration CLI with `--dry-run` and report output.
- Keep temporary legacy reader during transition.
- Log deprecation warnings before hard removal.

## Rollback

- Feature flag to switch consumers back to legacy reader temporarily.
- Keep dual-write mode until parity is confirmed.
