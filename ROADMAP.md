# Darkroom – Implementation Roadmap

**Goal**: produce a Lightroom Classic/CC-competitive desktop photo editor built on the
Darktable codebase, adding the major premium features that Lightroom has and Darktable
currently lacks, while keeping (and improving) Darktable's existing strengths.

---

## Current State (Darktable v5.x baseline)

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
- [x] **0.3** Docker multi-stage build with KasmVNC (browser GUI) + GPU passthrough
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

## Docker Deployment

### Files added in this repository

```
docker/
├── Dockerfile               # Multi-stage: builder (Ubuntu 24.04) + runtime (KasmVNC)
├── docker-compose.yml       # Profiles: default (CPU), gpu (NVIDIA), opencl (AMD/Intel)
└── kasmvnc-autostart.sh     # Launches Darkroom inside the VNC desktop
```

### GPU access

| GPU vendor | How to enable |
|---|---|
| NVIDIA | `docker compose --profile gpu up` (requires nvidia-container-toolkit) |
| AMD | `docker compose --profile opencl up` (requires ROCm driver on host) |
| Intel Arc / iGPU | `docker compose --profile opencl up` (requires intel-opencl-icd on host) |

### Quick start

```bash
# Build
docker build -t darkroom -f docker/Dockerfile .

# Run with NVIDIA GPU
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

**Total calendar time (single developer)**: ~44 weeks from today.
**With a 4-developer team**: ~14–16 weeks for phases 0–5 in parallel.

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
| NVIDIA Container Toolkit | GPU passthrough | Docker |

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
| GPU acceleration | Limited | ✅ OpenCL (NVIDIA/AMD/Intel) |
| Docker container | ❌ | ✅ Phase 0 |
| Open source | ❌ | ✅ |
| Cost | $9.99/month | Free |
| Privacy | Adobe cloud | Self-hosted, local-first |
