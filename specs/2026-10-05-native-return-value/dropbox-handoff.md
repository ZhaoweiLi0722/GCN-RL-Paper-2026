# Authorized Dropbox Handoff

Subsequent direct user instruction, after the single experiment approval:
`你要注意把我们这些实验的这个 artifacts都更新到那个 Dropbox里面。`

This supersedes the earlier no-Dropbox restriction for these experimental
artifacts only. It does not authorize messages, permission changes, other cloud
destinations, Git pushes, new experiments or modification of old archives.

Previously configured destination, confirmed to exist locally:
`/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/GCN-DRL Paper 2026/Research Artifacts`.

Use additive run-named subdirectories. Preserve completed and failed attempt
archives, inventories/receipts, protocols/authorization and terminal readouts.
Reuse existing archives instead of recompressing or re-auditing old science.
Do not export a changing partial file as final. The new native-return run's final
archive is copied after termination; a failure is preserved and exported too.

Write a manifest of source/destination/size/SHA256 and verify destination bytes.
Local Dropbox-folder copies, cloud-synchronized files and collaborator access
are separate facts. No cloud or access confirmation may be invented. Export
receipt/status will be recorded here or in a companion manifest. At this
authorization checkpoint no new copy has yet occurred.

## Verified Local Delivery

Completed2026-10-05T22:43:31Z under existing Research Artifacts/delivery_20261005.
The handoff covers Oct1-5 experimental artifacts, including closed unsuccessful
attempts and negative comparisons. It reuses25existing archives/manifests;
unarchived closed failures are copied as their preserved file trees, not
recompressed or scientifically re-evaluated. Protocols, authority/runtime/input/
seed locks, source/config/test/manuscript snapshot and launcher evidence included.
The currently running native-return result tree is explicitly excluded until
its final/failed boundary. Its approved protocol and frozen locks are included.

-18511source files verified:48028098688bytes (48.028decimalGB).
-18507new copies:40865583897bytes;4existing files reverified:7162514791bytes.
-Every destination SHA256 and length checked; archive bytes also match their
 existing manifest hash. Existing member-verification receipts reused.
-No different destination overwritten, no sharing-permission change or message.

Local export plan, per-file receipt and completion:
reports/dropbox_delivery_20261005_plan.json;
reports/dropbox_delivery_20261005_receipts.jsonl;
reports/dropbox_delivery_20261005_completion.json.
Copies of those three records are in delivery_20261005/handoff. Some earlier
archive files remain in their already-established sibling run directories;
the per-file receipts identify their exact existing destinations.

Status: local Dropbox-folder copy VERIFIED; cloud synchronization UNVERIFIED;
collaborator access UNVERIFIED. Attempting the Dropbox UI returned a locked-Mac
blocker, so no cloud completion claim is made. A later unlocked UI or actual
cloud receipt is needed to confirm off-device completion. Preserve this distinction
in all progress reports; do not recopy or rebuild unchanged successful artifacts.

Current launch/training handoff updates are delivered additively in
delivery_20261005/handoff/launch_update_20261005, with their own receipt.
The same active gcn-rl monitor must export the new run after termination,
including terminal failure if one occurs, before its final handoff/PAUSED state.
