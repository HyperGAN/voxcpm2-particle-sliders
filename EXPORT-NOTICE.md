# Sanitized public source snapshot

This public repository is a fresh source snapshot from the approved sanitized
export at commit `059d5159d617a47336495431beb602b913e07a97`. That export
contained 45 tracked files: 42 explicitly approved source paths plus three
export metadata files. SOURCE-PROVENANCE.json retains the original reviewed
source lineage, Git blobs and hashes; it records every further sanitation change
with the previous reviewed-export hash and the current exported hash.
The original private repository and its history were never read or copied for
this publication. Its implementation review remains pending; publication is not
approval of a training campaign or proof of native quality.

The only hardware-code changes replace workstation UUIDs with explicit dummy
UUIDs. The protected-GPU checks, cooperative /tmp/anima-1.lock lease protocol,
active-lease requirement, live process checks, memory floors and all scientific
admission guards remain. These placeholders admit no actual GPU by default.

Recordings, audio, model/adapter weights, judge maps or keys, evaluation requests,
private evidence, environments, run outputs and receipts are excluded. Templates
and the map-generation algorithm are source only. Do not publish their generated
private outputs. Historical source statements do not establish that no later
private training occurred. This export asserts only its own content and checks.

Historical dependencies remain core0.11/PG0.9.3 by default. The opt-in a794
core0.12/PG0.9.4 HNDL cohort stays separate and retains exact source admission.
Wide-waveform training remains refused; no reshape, shortening, alternate loss
or hidden-state fallback is added. Ordinary-LoRA distillation is approximate.
No pretrained speech, GPU fit, intelligibility, character fidelity or real-model
training/export qualification is established by this publication.

PUBLICATION.md records the current source-only review and CPU check scope.
No model weights, audio, paid calls, deployment or hardware admission is included.
Automated CI is not configured; no hosted CI pass is claimed.
