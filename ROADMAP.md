# C-41 — Roadmap unica

> **Documento canonico del piano.** Qui convergono roadmap di prodotto, parità con darktable, migrazione Rust/GTK4 e integrazione degli agenti AI. `PARITY_SPRINT_PLAN.md` è sostituito da questo documento; `RUST_MIGRATION_PLAN.md` resta solo un registro tecnico/storico. `PROGRESS.md` è il diario delle consegne e verifiche, mentre `PARITY_AUDIT.md` conserva il registro delle lacune/deviazioni osservate. `docs/parity-inventory/` contiene evidenze e matrici di capability a supporto di S9: non è una seconda roadmap e non decide priorità.
>
> **Obiettivo:** mantenere C-41 funzionante durante la riscrittura incrementale Rust/GTK4 e raggiungere la parità verificabile con una versione upstream fissata di darktable. Le sole esclusioni di prodotto approvate sono cloud sync e app mobile. In parallelo, offrire agli agenti AI locali un’interfaccia sicura e reversibile sullo stesso motore usato dalla UI.

## 1. Regole di scope e architettura

- Non unire darktable upstream in blocco. Per ogni delta decidere se adattare in Rust, applicare una correzione C/FFI solo quando il call path è provato, usare il comportamento come riferimento oppure escluderlo motivandolo.
- GTK3 e OpenCL upstream sono riferimenti, non codice da importare automaticamente: la UI di prodotto resta GTK4/Rust; il GPU backend C-41 è lavoro esplicito.
- La parità copre moduli e utility, import/export, formati e RAW, UI/workflow, parametri/default/storia/persistenza, colore/proofing, maschere, istanze multiple, Lua e GPU. Nessuna eccezione ulteriore senza decisione esplicita del proprietario.
- Preservare `/config/darkroom`, la tabella SQLite `darkroom_ui_prefs` e le variabili `DARKROOM_*`. Ogni migrazione futura di percorso o schema deve migrare i dati esistenti.
- Preview, export, undo/redo, persistenza e comportamento numerico vanno verificati insieme. Le differenze numeriche ammesse richiedono riferimento C e tolleranza esplicita.

## 2. Stato sintetico e baseline

- **Baseline darktable:** `97912518a5a9a9786aeee3c38116fbc999412ff8` (2026-11-13); merge-base `2939f5949aa17405f247859bb4fc272fda3b45c0`. Il confronto iniziale conta 2.379 commit upstream-only e 789 C-41-only: conteggi puntuali, non una proposta di merge.
- La ricognizione dei commit è stata **selettiva**, non esaustiva. Sono già emersi `contrastntexture`, `satcurvergb`, `spektrafilm`, DNG OpcodeList3 GainMap, modifiche RAW/white balance, highlights, precisione, HDR, maschere e cambiamenti GTK3/OpenCL. La prima sweep S9 è suddivisa in `docs/parity-inventory/modules.md`, `raw-formats.md`, `workflows-runtime.md` e nel ledger parziale `commit-triage.md`; sono matrici/evidenze conservative, non l’inventario comportamentale completo né il triage di tutti i commit.
- **Consegne di parità registrate S1–S8:** framing, chromatic aberrations, watermark, color calibration, prime slice di liquify e retouch, fondazione ICC e Rec.2020 per non-RAW. Non equivalgono a parità completa: le lacune residue sono elencate sotto.
- **Prossimo incremento formale: S9**, inventario upstream completo e prima ondata di import/porting. Un primo tentativo di Contrast & Texture è stato abbandonato prima del commit: il limite arbitrario di 4 milioni di pixel saltava silenziosamente l’effetto sulle immagini più grandi; rimuoverlo senza una strategia memoria/tiling coerente con il riferimento non è sicuro.
- Deploy corrente: `c-41:6b2ba54e4f` (GPU span fusion source `868a94e6`, full-C builder fix `ffbf7a65`, docs release `6b2ba54e4f`), healthy, zero restart; GPU passthrough è configurato e la GTX 1060/OpenCL ICD è visibile. `c-41-rollback-6b2ba54e4f` conserva il precedente live image `c-41:8acf0f13e0`. Le modifiche non verificate o non committate non sono parte delle release.

## 3. Filone A — Fondazioni e migrazione Rust/GTK4

Strategia: sostituire sottosistemi un passo alla volta dietro confini FFI stabili, mantenendo il prodotto eseguibile.

| Area | Stato / prossimo lavoro |
|---|---|
| Infrastruttura | Fork e mirror GitHub/Gitea, build Docker/KasmVNC e packaging di base esistono. Restano branding completo e icona, matrice CI Linux/macOS/Windows e pubblicazione GHCR da valutare/completare. L’immagine GPU `c-41:6b2ba54e4f` con span Exposure/Sigmoid è distribuita e verificata su NVIDIA; il container live vede la GTX 1060/OpenCL ICD. L’accelerazione resta parziale e il dispatch di immagini utente non è strumentato. AMD/Intel richiedono un runtime/ICD vendor nel container e non sono ancora verificati. |
| Pipeline immagini | Fase al confine Rust/FFI: i kernel portati continuano a crescere per incrementi; la copertura dei moduli C non è sinonimo di funzionalità UI o parità di prodotto. Requisito GPU ad **alta priorità**: span contigui Exposure/Sigmoid (solo se contengono Sigmoid) si eseguono con un dispatch OpenCL, fallback CPU e senza tiling; gli altri stage restano CPU. Differenziale passato nel container dell’immagine full-C live su GTX 1060; il dispatch di immagini utente nella preview/export live non è strumentato. |
| Database | `c41-db` fornisce CRUD per immagini, collezioni, tag, metadati e storia. Le nuove funzioni richiedono migrazioni compatibili e test di restart. |
| UI GTK4 | Lighttable, darkroom, pannelli, storia, import/export e funzioni correlate sono in sviluppo incrementale. Ogni controllo deve essere realmente collegato a pipeline, storia e persistenza. |
| Tecnica legacy C/FFI | Conservare solo dove necessario o conveniente; compilare le modifiche C con la build completa Release `-Werror`. Il binario Rust cargo-only deve restare senza dipendenze CMake non richieste. |

`RUST_MIGRATION_PLAN.md` contiene la ricognizione tecnica storica di kernel, loop e crate; le sue vecchie decisioni di scope sono superate da questo documento.

## 4. Filone B — Funzioni di prodotto pianificate

Queste voci vengono mantenute dalla roadmap precedente, ma non sono dichiarate completate né ordinate prima del lavoro di parità: priorità e dipendenze si aggiornano attraverso gli sprint qui sotto.

| Fase | Risultato richiesto |
|---|---|
| **P0 — Infrastruttura/branding** | Fork/mirror e Docker multi-stage con KasmVNC esistono; completare stringhe/nome/icona/splash/titolo, matrice CI Linux/macOS/Windows e publish GHCR. Mantenere opzionale il device passthrough e descrivere con precisione quali stage GPU sono supportati, senza implicare accelerazione dell’intera pipeline. Conservare percorsi e compatibilità dati. |
| **P1 — Smart Previews** | Tabella `smart_preview` nel catalogo (imgid, proxy JPEG/EXR, mipmap, timestamp); generazione collection-wide da CLI; sostituzione trasparente dell’originale offline; indicatore nelle miniature e preferenze qualità/dimensione. Quando l’originale torna disponibile, rimuovere il proxy e riapplicare la stessa storia. |
| **P2 — Copie virtuali** | Colonna `virtual_copy_of`/migrazione DB; copie condividono originale e film ma hanno `history` e `masks_history` indipendenti. Aggiungere “Create Virtual Copy”/“Promote to Master”, badge/stack in lighttable ed export singolo o di tutte le copie. |
| **P3 — Volti e People view** | Valutare InsightFace o YOLO-face su ONNX; job d’import; tabella faces (rettangolo, embedding, person_id); clustering cosine/DBSCAN e tabella persons; vista Persone, assegnazione/creazione persone, tag automatico solo dopo conferma. Tutti i dati restano locali. |
| **P4 — AI auto-tagging** | Valutare CLIP ViT-B/32 o MobileNet; task post-import; `suggested_tags(imgid, tag, confidence)`; pannello con suggerimenti selezionabili e CLI batch. Ricerca semantica è un’estensione da valutare, non prerequisito per i tag. |
| **P5 — Coda AI batch** | Tabella `ai_queue` con task/params/stato/priorità; pool di worker configurabile, progress reporting e precedenza al darkroom attivo; azioni lighttable per denoise/upscale/tag ed eventuale enhancement on-export. |
| **P6 — Template stampa** | Motore JSON per griglie, margini e campi; template contact sheet, 4×6, trittico e pagina fotolibro; editor drag/resize/snap; variabili titolo, data, camera e obiettivo. |

Le stime settimanali nella vecchia roadmap sono indicative e non sono un impegno temporale. Non esiste una stima credibile della fine della parità prima dell’inventario completo.

## 5. Filone C — Parità completa con darktable

### 5.1 Definizione di completamento

Parità significa coprire tutte le capacità utente e di elaborazione del baseline fissato. Il programma è **a baseline mobile**: ogni refresh crea un nuovo SHA, rigenera l’inventario, classifica ogni delta e pianifica le nuove differenze. Non basta avere una riga di catalogo o un kernel: servono UI utilizzabile, comportamento, persistenza, coerenza preview/export e test di riferimento.

### 5.2 Backlog per aree

- **C1 — Moduli visibili e interazione:** le righe storicamente inerti sono state collegate a slice di Framing, CACorrectRGB, Watermark, Color Calibration, Liquify e Retouch. Restano aperti, tra gli altri, controlli/asset completi di Watermark, canvas e interpolazioni di Liquify, multi-stroke/heal/masks/wavelets di Retouch e moduli nuovi del baseline.
- **C2 — Colore:** input/output profile e percorso non-RAW Rec.2020 sono parzialmente implementati. Restano soft proof, gamut-check, selezione/trasformazione display profile, adattamento alla presentazione finale sRGB e coerenza di thumbnail, full preview, stampa ed export.
- **C3 — Istanze e maschere:** stack di istanze indipendenti con parametri, blending, storia, identità UI, preset, serializzazione e ordine pipeline; poi tutte le famiglie/operazioni di maschera e le loro interazioni/persistenza.
- **C4 — Lua:** inventariare e implementare l’API Lua supportata dal baseline, lifecycle, eventi, storage/UI extension, test di compatibilità e sandbox.
- **C5 — GPU (priorità owner alta):** estendere il backend Rust OpenCL (in produzione: span contigui Exposure/Sigmoid contenenti Sigmoid) ad altri stage utili, con scelta runtime, fallback CPU, limiti memoria/tiling e test differenziali CPU↔GPU. La fusione è distribuita e il differenziale è passato sul runtime NVIDIA GTX 1060. Restano stato backend, scelta/soglia device, più stage residenti e tiling/spatial; AMD/Intel runtime e ICD vendor non sono distribuiti/verificati. Il solo accesso ai device o il backend OpenCL del C legacy non conta.
- **C6 — RAW, formati, risorse e workflow:** HEIC/HEIF/AVIF, X3F, GainMap DNG, forward matrices, demosaic/highlights, white balance, default/auto-enable, profili, preset, stili, `darktablerc`, metadati/XMP, storia e re-decode su undo/redo/jump, selezione, layout, unità, localizzazione, export/stampa e scorciatoie.
- **C7 — Registro deviazioni:** confrontare e risolvere ogni comportamento divergente annotato in `PARITY_AUDIT.md`; le deviazioni non possono diventare esclusioni implicite.

### 5.3 Sequenza sprint

**Priorità trasversale:** l’uso effettivo della GPU è requisito ad alta priorità; C5 resta anticipato rispetto agli sprint ordinari e prosegue in parallelo a S9. Il backend live (`c-41:6b2ba54e4f`) fonde gli stage Exposure con span Sigmoid contigui; differential hardware test su GTX 1060 passato nel full-C runtime. Mantenere fallback CPU/CI senza hardware e misurare trasferimenti/sincronizzazione per ogni estensione. Il dispatch live di foto utente non è strumentato; test e limiti sono in `PROGRESS.md`.

| Sprint | Obiettivo e criterio di uscita |
|---|---|
| **S1–S4 — consegnati** | Framing, Chromatic Aberrations, Watermark e Color Calibration collegati ai rispettivi percorsi. Restano aperte slice di parità completa indicate in C1/C7. |
| **S5–S7 — consegnati a slice** | Liquify con lattice UI; Retouch con singolo clone stamp e feather. La canvas libera e le funzioni avanzate restano aperte. |
| **S8a–S8b — consegnati** | Fondazione ICC e processing/export Rec.2020 per fonti non-RAW idonee; preview visibile resta sRGB. Proof/display completo è ancora S10. |
| **S9 — in corso** | Completare inventario capability/delta per lo SHA fissato; le prime matrici per moduli, RAW/formati e workflow/runtime sono in `docs/parity-inventory/`. Verificare le classificazioni conservative, triagiare i commit (PORT, APPLY, REFERENCE o non applicabile con motivazione), assegnare priorità e sprint; il fallback rawprepare è il primo slice portato. Nessuna sweep è completa finché ogni riga ha evidenza e verifica. |
| **S10** | Colore/proofing completo (C2). |
| **S11** | Architettura multi-istanza e migrazioni (C3). |
| **S12** | Maschere complete, dipende da S11 (C3). |
| **S13** | Nuovi moduli upstream: Contrast & Texture, RGB Saturation Curve, Spektrafilm e inventario aggiornato. Audit algoritmi, dati/licenze, reference e memoria prima di ciascun port. |
| **S14** | RAW, formati, sensori e metadati (C6). |
| **S15** | Compatibilità Lua e sandbox (C4). |
| **S16 — priorità alta, anticipato/in corso** | Span Exposure/Sigmoid contigui dispatchati con un singolo upload/kernel/readback full-frame; deployed in `c-41:6b2ba54e4f`, differential GPU test passed su NVIDIA GTX 1060 e fallback CPU conservato. Resta da strumentare il dispatch con foto utente. Poi: esporre stato backend, valutare soglia d’uso e allargare a più stage compatibili/residenti; spatial/tiling. AMD/Intel richiedono runtime/ICD nel container. Il solo device passthrough e OpenCL C legacy non soddisfano il gate. |
| **S17** | Workflow, UI, configurazione, preset e stampa (C6). |
| **S18** | Chiudere il registro delle deviazioni (C7). |
| **S19 — ricorrente** | Aggiornare SHA upstream e rigenerare inventario/delta. |
| **S20 — gate di release** | Suite differenziale e UX, sicurezza agenti, CI completa, dual push e deploy; ripetere sul baseline successivo. |

### 5.4 Prima triage upstream (campione, non inventario completo)

La prima analisi ha evidenziato: `1067c653c0`/`23d5544c8d` (Contrast & Texture); `92c0f79a83`/`7076cab103` (RGB Saturation Curve e fix); `d81ccfe23b` e follow-up Spektrafilm; `c7d98b58ea`, `dcb52f7ffe`, `26117d3114` (DNG GainMap); cambiamenti white-balance/manual calibration; `56b0ca0081` (fallback rawprepare quando white ≤ black; portato in `rawimage.rs` nel primo slice S9, con test Bayer/X-Trans/override); fix highlight; e semantica HDR/PQ/HLG. I cluster `satcurvergb` e Spektrafilm sono verificati in `docs/parity-inventory/commit-triage.md` (**PORT, S13**); DNG OpcodeList3 GainMap, ForwardMatrix/HueSatMap e white balance/manual calibration hanno triage parziale **PORT/REFERENCE, S14**. Il registro documenta call path e sovrapposizioni non equivalenti, gate dati/licenza/memoria per Spektrafilm, parser OpcodeList2/helper Bayer e OpcodeList3 warp/vignette nel C legacy (non GainMap), lettura senza applicazione delle ForwardMatrix in C-41 e distinzione fra WB Rust e IOP C legacy. Le liste di follow-up non sono exhaustive; gli altri cluster citati restano da verificare e classificare. Le matrici non sono classificazioni definitive: ogni elemento va confrontato con il call path e la versione C-41, non cherry-picked alla cieca. Il ledger deve includere SHA, feature, decisione, motivazione, sprint e verifica.

## 6. Filone D — Agent-native editing, Phase 9

MCP locale su **stdio** è il primo adattatore; il contratto stabile è una API Rust versionata e indipendente dal protocollo. Niente listener di rete, shell/eval, accesso arbitrario al filesystem o manipolazione GTK in v1.

1. **9a — Sessione headless condivisa:** isolare apertura immagine, edit corrente, rendering preview, capacità, validazione patch, storia, undo ed export dal codice GTK. UI e agenti devono chiamare lo stesso servizio; schema pubblico tipizzato, mai byte grezzi `PreviewParams`.
2. **9b — MCP read-only:** ricerca catalogo, metadati, edit corrente, capacità e preview limitata; risultati strutturati e client interoperabile.
3. **9c — Edit reversibile:** `plan_edit`, `preview_edit`, `commit_edit`, diff chiaro, allowlist testata, commit atomico come un passo di storia e undo/redo.
4. **9d — Export derivato confinato:** root configurata in C-41, creazione race-resistant senza symlink/path escape o sovrascrittura, limiti risorse e mai scrittura sull’originale.
5. **9e — Approvazione e audit:** attribuzione visibile, revisione umana, persistenza, rifiuto permessi, test end-to-end e guida d’uso.

**Invarianti:** sola lettura di default; permessi espliciti per scritture/export; targeting tramite ID catalogo; patch validate contro schema e revisione corrente; token ottimistico condiviso tra UI e agenti; commit stale rifiutato e ri-pianificato; limiti preview/job; fail-closed quando serve approvazione.

**Uscita:** un client MCP supportato può scoprire immagini e controlli, ispezionare un edit, chiedere preview bounded, proporre una modifica, vedere prima/dopo, committare solo se autorizzato, ritrovare l’edit dopo restart e annullarlo dalla UI ordinaria; export opzionale confinato.

## 7. Verifica, rilascio e tracciamento

Per ogni incremento, senza scorciatoie:

1. Delegare lo sviluppo pesante a un agente generalista con file/scope e mandato read-only sul resto.
2. Far eseguire una review indipendente a un agente senior distinto dallo sviluppatore; correggere tutti i rilievi.
3. Eseguire `scripts/ci-local.sh` in Docker e fidarsi dei codici di uscita: workspace check, Clippy, test workspace `--release`, link del binario Rust.
4. Per C, aggiungere la build full-app Release `-Werror`; per l’immagine usare `CACHEBUST` o `--no-cache`.
5. Eseguire `git diff --check`; aggiungere a `PROGRESS.md` una voce UTC con cambiamento, verifiche, errori e correzioni. Aggiornare `PARITY_AUDIT.md` nello stesso commit quando cambia il registro.
6. Commit deliberato, push a GitHub e Gitea, verifica di entrambi i ref e dei check CI.
7. Deploy al container `c-41` quando appropriato, smoke test e conservazione del rollback. Cambiare solo quel servizio senza autorizzazione.
8. Nessun segreto in task, diff, log, progress o commit. `install-debuntu.sh` e `.sha256sum` restano untracked e intatti.

**Fonti di stato:** questo documento è l’unica roadmap. `PROGRESS.md` registra le consegne, `PARITY_AUDIT.md` i dettagli delle lacune/deviazioni, `RUST_MIGRATION_PLAN.md` i dettagli tecnici storici. A ogni sessione riprendere da roadmap → coda di `PROGRESS.md` → sezioni pertinenti dell’audit.
