# VoxMatrix rename compatibility

VoxMatrix is the canonical project and distribution name. Compatibility
surfaces are retained so existing Registry entries, manifests, and automation
do not need to migrate atomically.

| Surface | Canonical | Compatibility |
| --- | --- | --- |
| Distribution | `voxmatrix` | no second distribution is bundled |
| Main CLI | `voxmatrix` | `ultraeval-audio` |
| Dataset preparation | `voxmatrix-prepare` | `ultraeval-audio-prepare` |
| Shared session | `voxmatrix-suite` | `ultraeval-audio-suite` |
| Public facade | `import voxmatrix` | `audio_evals`, `mesh_eval` |
| Environment variables | `VOXMATRIX_*` | matching `ULTRAEVAL_*` names |

When both environment-variable forms exist, `VOXMATRIX_*` takes precedence.
Serialized Python class paths under `audio_evals` and `mesh_eval` remain
stable because Registry YAML and historical manifests may contain them.

Historical manifests using the non-standard splits `replay_mini` or
`event_reuse_mini` are normalized to `test`; the original value is kept in
`source_split`.

The UltraEval-Audio paper title, citation, upstream URL, and provenance records
retain their original spelling. New code, documentation, configuration, and
environment variables should use VoxMatrix naming.
