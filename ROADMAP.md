# Darkroom – Implementation Roadmap

**Goal**: produce a Lightroom Classic/CC-competitive desktop photo editor built on the
Darktable codebase, adding the major premium features that Lightroom has and Darktable
currently lacks, while keeping (and improving) Darktable's existing strengths. C-41
must also offer a safe, documented tool interface so external AI agents can inspect,
preview, and make reversible photo edits without screen-scraping or driving GTK.

---

## Upstream Current State (Darktable v5.x baseline — not the C-41 GTK4 application)

Darktable already covers the core RAW-editing workflow at a level that rivals or
exceeds Lightroom in several areas:

| Category | Darktable today |
|---|---|
| RAW demosaicing & processing | ✅ Full (RawSpeed + LibRaw, 106 IOP modules) |
| Non-destructive parametric editing | ✅ Superior (fully modular pipeline) |
| Filmic / AGX / sigmoid tone mapping | ✅ Best-in-class |
| AI object masking (SAM2.1 / SegNext) | ✅ Exceeds Lightroom |
| AI denoise / super-resolution | ✅ (BSRGAN, NAFNet, NIND) |
| OpenCL GPU acceleration | ✅ NVIDIA / AMD / Intel |
| Tethering via libgphoto2 | ✅ |
| Lua 5.4 scripting | ✅ |
| Formats: AVIF, JXL, HEIC, EXR, DNG … | ✅ |

**Primary gaps vs Lightroom** (ordered by user impact):

1. Face detection & recognition
2. Cloud sync / remote catalog
3. Smart Previews (offline editing without originals)
4. Mobile companion app (or web remote)
5. Virtual Copies with independent histories
6. AI auto-tagging / keyword suggestions
7. Batch AI processing (denoise / upscale on queue)
8. People / People-groups view
9. Print layout templates (multi-image layouts)
10. Docker / container deployment

---

## Milestones

### Phase 0 – Infrastructure (weeks 1–4)

Prerequisite work; no user-visible features.

- [x] **0.1** Fork darktable-org/darktable → ilfrick/c-41 on GitHub
- [x] **0.2** Mirror to self-hosted Gitea (www.housefz.com/git/ilfrick/c-41)
- [x] **0.3** Docker multi-stage build with KasmVNC (browser GUI) + optional GPU device passthrough (the launched C-41 GTK4 app remains CPU-only)
- [ ] **0.4** Rename project string "darktable" → "Darkroom" in UI, about dialog,
       config paths (`~/.config/darkroom`, `~/.cache/darkroom`)
- [ ] **0.5** CI pipeline: GitHub Actions build matrix (Linux/macOS/Windows) +
       Docker image publish to GHCR
- [ ] **0.6** Branding: new app icon, splash screen, window title

---

### Phase 1 – Smart Previews (weeks 5–10)

**Why first**: unblocks offline/mobile workflows and is self-contained.

**Lightroom feature**: JPEG proxies embedded in the catalog that allow full editing
without the original RAW file present (useful for travel, laptop editing).

**Implementation**:
- Add `smart_preview` table to `library.db` (imgid, embedded JPEG or EXR, mipmap
  level, generation timestamp).
- New IOP: `smart_preview_source` — transparently substitutes original with proxy
  when original path is unavailable.
- `darkroom-cli` export mode: `--generate-smart-previews [--size 2560]` builds
  proxies for an entire collection.
- UI indicator in lighttable thumbnails: lock icon when working off proxy.
- Sync logic: if original becomes available, discard proxy and re-apply the same
  edit history without re-editing.

**Files to create/modify**:
- `src/common/smart_previews.{h,c}` – proxy management API
- `src/iop/smart_preview_source.c` – proxy IOP
- `src/common/collection.c` – add offline filter
- `src/gui/preferences.c` – proxy quality / size settings
- `data/ui/smart_preview_prefs.ui`

---

### Phase 2 – Virtual Copies (weeks 8–14)

**Why**: directly requested by Lightroom power users; relatively self-contained
change to the database schema.

**Lightroom feature**: multiple independent edit histories per physical file,
each showing as a separate thumbnail.

**Implementation**:
- Current darktable "duplicates" share the same `images` row with a `group_id`.
  True virtual copies need an independent `history` stack per copy.
- Schema change: add `virtual_copy_of` column to `images`; copies share `film_id`
  and `filename` but have independent `history` and `masks_history`.
- Context menu: "Create Virtual Copy", "Promote to Master".
- Lighttable: stack badge shows copy count; expanding a stack shows all copies.
- Export: optional "export all copies" mode.

**Files**:
- `src/common/image.c` / `image.h` – add virtual copy fields
- `src/common/history.c` – copy isolation
- `src/gui/lib/filmstrip.c` – stack UI
- Database migration script: `src/common/database.c` (schema version bump)

---

### Phase 3 – Face Detection & People View (weeks 12–22)

**Why**: the highest-profile missing feature vs Lightroom.

**Implementation**:

#### 3a – Face Detection (weeks 12–16)
- Integrate **InsightFace** (ONNX model, fits the existing AI ONNX backend in
  `src/ai/`), or alternatively **YOLOv8-face** for lighter weight.
- New AI task type `FACE_DETECT` alongside existing segmentation/denoise tasks.
- Detection runs as background job on import; results stored in `faces` table:
  `(id, imgid, rect_x, rect_y, rect_w, rect_h, embedding BLOB, person_id)`.
- Drawn face-mask overlay in darkroom: detected faces show as bounding boxes,
  clickable to apply a parametric or drawn mask scoped to the face region.

#### 3b – Face Recognition & People View (weeks 17–22)
- Cluster face embeddings (cosine similarity, DBSCAN) → `persons` table.
- Lighttable "People" view: grid of person clusters, similar to LR's People view.
- UI: drag a face crop onto a person to confirm; click "Unknown" to name a new
  person.
- Keyword integration: confirmed person → auto-tag image with person's name.
- Privacy: all embeddings stored locally; no cloud upload ever.

**New files**:
- `src/common/faces.{h,c}` – detection, storage, clustering
- `src/ai/face_detect.{h,c}` – ONNX inference wrapper
- `src/views/people.{h,c}` – new lighttable view
- `src/libs/people_panel.{h,c}` – panel widget

---

### Phase 4 – AI Auto-Tagging (weeks 18–24)

**Lightroom feature**: AI-suggested keywords based on image content (places, objects,
people).

**Implementation**:
- Integrate **CLIP** (ViT-B/32 ONNX) or **MobileNet-v3** for image classification.
- New AI task `AUTO_TAG`: runs post-import, produces ranked keyword suggestions
  stored in `suggested_tags(imgid, tag, confidence)`.
- Lighttable: "AI Suggestions" panel shows top-5 tags with checkboxes; accept to
  write to `tags` table.
- Batch: `darkroom-cli --auto-tag [collection]`
- Optional: CLIP-powered semantic image search ("find images with mountains at
  sunset") via a search bar in the collection module.

---

### Phase 5 – Batch AI Processing (weeks 20–26)

**Why**: Lightroom users expect to apply denoise/upscale to hundreds of images at
once; the existing AI backend is single-image only.

**Implementation**:
- New `ai_queue` table: `(id, imgid, task_type, params JSON, status, priority)`.
- `src/common/ai_queue.{h,c}`: enqueue/dequeue, worker thread pool (configurable
  N workers), progress reporting via existing progress-bar API.
- Lighttable context menu: "Send to AI Queue → [Denoise / 2× Upscale / 4× Upscale
  / Auto-tag]".
- Processing priority: active darkroom image always pre-empts queue jobs.
- Export integration: "Apply AI enhancement on export" option (on-the-fly, no queue).

---

### Phase 6 – Cloud Sync & Remote Catalog (weeks 24–36)

**Why**: the largest architectural gap; requires a new backend service.

**Scope for v1**: catalog metadata + edit history sync (not raw pixel sync).

**Architecture**:

```
Darkroom Desktop  ←──→  Darkroom Sync Server  ←──→  Darkroom Desktop / Web
                          (REST + WebSocket)
                          SQLite → PostgreSQL
```

**Components**:

#### 6a – Sync Server (`darkroom-sync/`)
- Language: Go (small binary, easy to self-host or run in the companion Docker
  compose profile).
- API: REST for catalog CRUD, WebSocket for real-time push notifications.
- Auth: OAuth 2.0 / local accounts (Gitea-style).
- Storage: PostgreSQL or SQLite for small installs.
- Self-hosted first; cloud-hosted option (darkroom.app) later.

#### 6b – Client Sync Engine (`src/sync/`)
- `src/sync/client.{h,c}`: HTTP client (libcurl), conflict resolution (LWW or
  three-way merge on history stacks).
- Conflict UI: when two devices edited the same image, show diff of history stacks
  and let user pick.
- Sync scope selector: "This collection", "All rated ≥ 3★", "Everything".
- Bandwidth throttle, pause/resume.

#### 6c – Web Viewer (separate repo: `darkroom-web`)
- Read-only web gallery rendered server-side from synced catalog + cached JPEGs.
- Technology: SvelteKit + a small Rust service that renders edits via the
  darktable pixel pipeline compiled to WASM (stretch goal) or pre-rendered
  on the server.

---

### Phase 7 – Mobile Companion App (weeks 30–42)

**Scope**: iOS + Android app for browsing, rating, and basic adjustments that sync
via the Phase 6 server.

**Stack**: Flutter (single codebase, good camera API access).

**Features v1**:
- Browse synced catalog, view full-res previews.
- Set star ratings, color labels, flags.
- Write/read metadata: caption, keywords, GPS.
- Send to AI queue (trigger server-side processing).
- Capture to catalog: import photos from phone camera roll.

**Features v2**:
- Basic tone adjustments (exposure, contrast, white balance) synced back to desktop.
- Push notifications: "AI processing complete".

---

### Phase 8 – Print Layout Templates (weeks 36–44)

**Why**: the existing print view in darktable supports single-image output; Lightroom
has multi-image grid layouts with fine typography.

**Implementation**:
- Extend `src/views/print.c` with a layout engine: JSON template format describing
  cell grids, margins, text fields.
- Bundled templates: contact sheet, 4×6 grid, triptych, photo book page.
- Template editor: drag-and-drop cells, resize, snap to grid.
- Variables in text fields: `{title}`, `{date}`, `{camera}`, `{lens}`.

---

### Phase 9 – Agent-native photo editing (priority track; begin after the current UI/engine foundations)

**Goal**: let an external AI agent use C-41 as a photo-editing tool through a
stable, model-agnostic interface. The agent should discover supported controls,
propose edits, inspect a preview, and—when authorized—save a reversible edit or
export a derivative. It must not automate GTK or infer undocumented binary
parameter blobs.

**Product decision**: start with a local **MCP server over stdio** for
compatibility with MCP-capable agent hosts. Keep the protocol adapter thin over a
versioned C-41 Rust editing API; the API, not MCP, is the stable product
contract. No network listener, shell/eval tool, cloud account, or arbitrary
filesystem access in v1. A non-MCP host can later use the same API through a
small CLI/JSON-RPC adapter.

**Phases and deliverables**:

1. **9a — Headless editing contract.** First map ownership of preview,
   history, export, and catalogue writes. Extract a GTK-independent image-session
   service from the current `c41-ui` flows (candidate `c41-engine`/`c41-session`
   crate; name TBD) and have both UI and agent call it—do not duplicate rendering
   or introduce a second independent edit writer. Separate image-session
   operations (open a catalogue image, load its current edit, render a bounded
   preview, inspect capabilities) from GTK widget callbacks; define versioned
   contracts for patch validation, history, undo, and export before implementing
   them in later slices. Publish a versioned capability/parameter schema derived
   only from controls actually wired in C-41. Use stable module/parameter names,
   typed values, limits, defaults, and clear unsupported-operation errors; never
   expose raw `PreviewParams` bytes as the API.
2. **9b — Read-only MCP MVP.** Implement tools to list/search catalogue images,
   inspect metadata and current edit, enumerate supported modules/parameters,
   and render a small before/current preview. Tools return bounded structured
   results with image IDs and provenance; large image data is not returned
   inline. Verify interoperability with at least one MCP client.
3. **9c — Reversible edit loop.** Add `plan_edit` / `preview_edit` /
   `commit_edit` operations for a validated parameter patch. Return a concise
   diff and preview before persistence. Commit atomically as one named history
   step, preserve undo/redo, and report exactly which requested changes were
   accepted, clamped, or refused. Start with a deliberately small allowlist of
   well-tested live modules; expand only when preview/export/history parity is
   covered by tests.
4. **9d — Safe derivative export.** Require an output directory explicitly
   configured by the user in C-41; do not accept an agent-supplied root. Resolve
   the configured root and use race-resistant descriptor-relative creation
   (no-follow + create-exclusive semantics) so path traversal, symlink escapes,
   and replacement of an existing destination are refused. Also reject
   unsupported formats and resource-limit violations. Never modify or delete
   the source photo. Require
   explicit user approval for persistent edits and exports by default; provide a
   narrowly scoped opt-in policy for unattended local workflows.
5. **9e — Human-in-the-loop and quality gate.** Show agent-authored changes in
   the C-41 UI with source/agent attribution, editable history labels, preview,
   undo, and an obvious approval/rejection path. Add end-to-end tests for
   tool-schema compatibility, restart persistence, undo, path confinement,
   denied permissions, and representative edit tasks; document setup and a
   sample agent workflow.

**Security and trust invariants**: read-only is the initial/default mode; each
write/export is an explicit capability, not implied by opening a catalogue.
Target images by catalogue ID from user-configured roots, validate every patch
against the published schema and current image state, cap preview dimensions and
job resources, and log the agent identity plus accepted edit. Use an optimistic
edit revision token shared by every writer (GTK and agent); check the token and
persist the edit atomically in one transaction. If the UI or another client
changes the image after the agent preview, reject the stale commit and require
re-planning rather than clobbering either edit; the UI must refresh/reopen
external commits safely. No
arbitrary code, shell commands, raw-file overwrite, delete, unbounded path, or
silent persistence. Human approval cannot depend on an MCP host feature that may
not exist: the C-41 approval policy must work locally and fail closed when
required.

**Exit criteria**: a supported MCP host can discover C-41 capabilities, find an
image, inspect its current edit, request a bounded preview, propose a valid
multi-parameter edit, obtain a before/after preview, and commit it only with
permission; the saved history survives restart, can be undone in the ordinary
UI, and an optional derivative export is confined to the configured destination.
A human can inspect and correct the agent's result using the regular editor.

---

## Docker Deployment

### Files added in this repository

```
docker/
├── Dockerfile               # Multi-stage: builder (Ubuntu 24.04) + runtime (KasmVNC)
├── docker-compose.yml       # Profiles: default (CPU), nvidia, amd, intel (device passthrough; c41-rs remains CPU-only)
└── kasmvnc-autostart.sh     # Launches Darkroom inside the VNC desktop
```

### GPU access

| GPU vendor | How to enable |
|---|---|
| NVIDIA | `docker compose -f docker/docker-compose.yml --profile nvidia up darkroom-nvidia` (requires nvidia-container-toolkit) |
| AMD | `docker compose -f docker/docker-compose.yml --profile amd up darkroom-amd` (requires ROCm driver on host) |
| Intel Arc / iGPU | `docker compose -f docker/docker-compose.yml --profile intel up darkroom-intel` (requires intel-opencl-icd on host) |

These profiles pass GPU devices into the opt-in full-C image. The launched
`c41-rs` GTK4 application is currently CPU-only; the OpenCL kernels in the C
code are not used by that application. GPU acceleration is therefore not yet
available in the C-41 product, despite the legacy C build being included in
that image.

### Quick start

```bash
# Build
docker build -t darkroom -f docker/Dockerfile .

# Run with NVIDIA device passthrough (does not accelerate the C-41 GTK4 app yet)
docker run --gpus all \
  -e PUID=$(id -u) -e PGID=$(id -g) \
  -p 3000:3000 \
  -v ~/Pictures:/photos \
  -v ~/.config/darkroom-docker:/config \
  darkroom

# Open browser at http://localhost:3000
```

---

## Effort Estimates

| Phase | Feature | Complexity | Weeks |
|---|---|---|---|
| 0 | Infrastructure, branding, CI | Low | 1–4 |
| 1 | Smart Previews | Medium | 5–10 |
| 2 | Virtual Copies | Medium | 8–14 |
| 3 | Face Detection + People View | High | 12–22 |
| 4 | AI Auto-Tagging | Medium | 18–24 |
| 5 | Batch AI Queue | Medium | 20–26 |
| 6 | Cloud Sync + Web Viewer | Very High | 24–36 |
| 7 | Mobile App | High | 30–42 |
| 8 | Print Layouts | Low | 36–44 |
| 9a–9e | Agent-native editing API + MCP tools | High; staged, gated by headless edit contract | Priority track; schedule after UI/engine foundations |

**Baseline estimate for phases 0–8 (single developer)**: ~44 weeks from the
original plan date; Phase 9 is a separately prioritized track and is not included
in that estimate. **With a 4-developer team**: ~14–16 weeks for phases 0–5 in
parallel.

---

## Technology Dependencies to Add

| Dependency | Purpose | Phase |
|---|---|---|
| InsightFace / YOLOv8-face (ONNX) | Face detection | 3 |
| CLIP ViT-B/32 (ONNX) | Auto-tagging, semantic search | 4 |
| Go 1.22+ | Sync server | 6 |
| PostgreSQL 16 (optional) | Cloud catalog backend | 6 |
| Flutter 3.x | Mobile app | 7 |
| KasmVNC | Browser-accessible desktop in Docker | Docker |
| NVIDIA Container Toolkit | Optional device passthrough (no C-41 GPU compute yet) | Docker |
| MCP Rust SDK / protocol adapter (evaluate API maturity and licensing first) | Local stdio tool server for AI-agent hosts | 9b |

---

## Competitive Positioning

| Feature | Lightroom CC | Darkroom (target) |
|---|---|---|
| Non-destructive RAW editing | ✅ | ✅ |
| AI object masking | Basic | ✅ Superior (SAM2.1) |
| AI denoise / upscale | ✅ (Denoise AI) | ✅ (NAFNet, BSRGAN) |
| Face detection | ✅ | ✅ Phase 3 |
| Smart Previews | ✅ | ✅ Phase 1 |
| Virtual Copies | ✅ | ✅ Phase 2 |
| Cloud sync | ✅ (Adobe) | ✅ self-hosted Phase 6 |
| Mobile app | ✅ | ✅ Phase 7 |
| GPU acceleration | Limited | Future target — not currently available in C-41; OpenCL (NVIDIA/AMD/Intel) |
| Docker container | ❌ | ✅ Phase 0 |
| Open source | ❌ | ✅ |
| Cost | $9.99/month | Free |
| Privacy | Adobe cloud | Self-hosted, local-first |
