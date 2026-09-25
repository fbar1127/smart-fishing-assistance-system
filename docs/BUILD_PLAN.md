# IsdaYou — Two-Person Build Plan (26 Sep → 20 Oct 2026)

**Builders:** Troy (main PC: i5-12th gen, 16 GB, Android Studio) and Diana (laptop, 16 GB).
**Test phone:** Troy's Infinix Zero 30 5G (USB debugging). **Research/content:** Jake and Franc.
**Rule of thumb:** always keep a version that runs. Each step ends with a working app on the phone and a commit.

## Big decisions (and why)

| Decision | Why |
|---|---|
| Troy builds the Android core; Diana builds data, backend, and the log/sync-upload parts in her own folders | Two people never edit the same file, so Git merges stay clean. |
| **Supabase** (hosted PostgreSQL + PostGIS) instead of self-hosting FastAPI + Docker | No server to write or keep running; PostGIS stays; its dashboard is the admin console. Saves about a week. |
| Zones are drawn from official maps by an admin (geojson.io → SQL template); no automatic scraper for the demo | Boundaries cannot be scraped reliably. A scraper for text advisories is optional (Step 7). |
| osmdroid (OpenStreetMap) for the map | No API key or billing needed. |

Manuscript sections to update because of these decisions: 14.1 (deployment stack), the scraper wording in 5.3, and the admin console description. Troy asks Claude to make these tracked edits.

## Timeline

| Week | Troy (Android core) | Diana (data + backend + log) | Jake & Franc (no device) |
|---|---|---|---|
| **1** · 26 Sep–2 Oct | Step 0 setup · **Step 1 fish ID on phone** | Step 0 setup · Supabase project + schema · species.json filled (Step 2) | Research the 31 species (local names, description, habitat, advisories) · find official zone maps (LLDA ZOMAP, BFAR, LGU sanctuaries) |
| **2** · 3–9 Oct | **Step 3 map + GPS + offline zone check** | **Step 4 catch & identification log (Room)** · encode zones in Supabase | Write test cases (Section 11/12) · recruit 3–5 fishers or classmates for usability testing |
| **3** · 10–16 Oct | **Step 5a download sync** · Step 6 zone alerts + blur check · phone speed/memory measurement | **Step 5b upload sync + consent screen** · advisories data | Draft manuscript Sections 12–22 from the plan and results |
| **Final** · 17–20 Oct | **Feature freeze 17 Oct** · bug fixes · airplane-mode test · build demo APK | Test every screen on the phone · fix data | Usability test sessions (with Troy's phone) · slides |

If a week slips, drop from the bottom: Step 7 first, then Step 6, never Steps 1–4.

---

## Step 0 — Setup (both, day 1–2)

**Troy**
1. Install **Git**, **GitHub Desktop** and **Android Studio** (latest stable). In Android Studio: SDK Manager → install Android 15 (API 35) and Android 9 (API 28).
2. Phone: Settings → About phone → tap *Build number* 7 times → Developer options → turn on **USB debugging**. Plug it in and accept the prompt.
3. GitHub Desktop → *Clone repository* → `fbar1127/smart-fishing-assistance-system`.
4. Copy `model_fp16.tflite` from Drive `IsdaYou/results_run1/efficientnet_lite0/seed_42/` into `mobile-app/app/src/main/assets/` and rename it `species_classifier.tflite`. Compare `labels.txt` from the same Drive folder with the one in assets (they must match line by line).
5. Android Studio → *Open* → select the `mobile-app` folder → wait for Gradle sync → ▶ Run on the phone.
6. Open the Claude desktop app → Code → choose the repo folder, so Claude can build and fix errors on your PC.

**Diana**
1. Install Git, GitHub Desktop, VS Code, and Python 3.11. (Android Studio is optional for her; she can run the app on the emulator for Step 4.)
2. Accept the collaborator invite; clone the repo.
3. Create a free project on **supabase.com** (region: Singapore). SQL Editor → paste `backend/supabase/schema.sql` → Run.
4. Project Settings → API: copy the **Project URL** and **anon public key** and send them to Troy privately (not in Git).

**Git rules:** create a branch per step (`step-1-fish-id`), commit when it works, open a Pull Request, the other person approves, merge. Nobody pushes straight to `main`.

---

## Step 1 — Offline fish identification (Troy) · FR fish ID, MLR-001–008, NFR-001

Already written in the starter: `FishClassifier.kt`, `Prediction.kt`, `IdentifyScreen.kt`.

**Prompt for Claude:** *"Read CLAUDE.md. Build the app and fix any compile errors without changing the rules. Then run the unit tests."*

**Test on the phone:** take 10 photos of fish images shown on a second screen (or pick dataset test images copied to the phone). Check: the name, confidence, alternatives, "Not sure" for a non-fish photo, and that it works in **airplane mode**. Note the inference time shown (target ≤ 3 s; expect well under 1 s).

## Step 2 — Species library content (Diana, with Jake & Franc) · BR03

Fill `mobile-app/app/src/main/assets/species.json` (31 entries): scientific name, local names (e.g., bangus, dalag, tilapya), short description, habitat, average size, advisory text for Knifefish and Janitor Fish, and a source for each. Keep `index` and `commonName` exactly as they are. Then run `python backend/supabase/seed_species.py > seed.sql` and run `seed.sql` in Supabase.

**Test:** the Library tab lists 31 species, search by local name works, and an identified invasive species shows the red advisory.

## Step 3 — Map, GPS and offline zone check (Troy) · FR-014–016, NFR-010

**Prompt:** *"Read CLAUDE.md. Step 3: add a Room database with a Zone table (id, name, zoneType, authority, ruleText, geometry GeoJSON text, activeFrom, activeTo, isActive, updatedAt). Seed it from a bundled `assets/zones.geojson` so it works offline before any sync. Add a Map tab using osmdroid centered on Laguna de Bay that draws the zones and the user's location (Fused Location Provider, ask permission). Show a red banner 'Inside: <zone name> — <rule>' when the current location is inside an active zone, using geo/PointInPolygon."*

**Test:** Diana adds a TEST zone around Troy's house; walk in and out of it with **airplane mode on**.

## Step 4 — Catch and identification log (Diana) · FR catch logging, BR-log

**Prompt (on Diana's laptop):** *"Read CLAUDE.md. Step 4, only in package `log/`: Room entities IdentificationLog (uuid, speciesIndex, confidence, top3 JSON, confident, lat, lng, createdAt, uploaded=false) and CatchLog (uuid, speciesIndex, quantity, weightKg, lat, lng, caughtAt, notes, uploaded=false), DAOs, and a Log screen listing both with an 'Add catch' form. Expose a `LogRepository.saveIdentification(prediction)` function that Troy will call from IdentifyScreen."*

Then Troy adds the one call in `IdentifyScreen` and the Log tab in `MainActivity` (his files).

## Step 5 — Sync (5a Troy download, 5b Diana upload) · Section 5.3, FR-024, MLR-016

- **5a Troy:** *"Step 5a: a WorkManager job that, when online, downloads rows changed since the last sync from Supabase REST (`/rest/v1/zones_geojson`, `/species`, `/advisories`, filter `updated_at=gt.<last>`) into Room. URL and anon key from local.properties via BuildConfig."*
- **5b Diana:** *"Step 5b: a Settings screen with a 'Share my logs to help improve IsdaYou' switch (off by default) and a WorkManager job that uploads logs with uploaded=false to `/rest/v1/identifications` and `/rest/v1/catch_logs` only when the switch is on, then marks them uploaded."*

**Test:** change a zone's rule text in the Supabase dashboard → go online → the phone shows the new text; turn sharing on → rows appear in Supabase; turn it off → nothing uploads.

## Step 6 — Zone alerts and photo quality (Troy) · FR-016, Section 11.6

**Prompt:** *"Step 6: (a) register Android geofences (circles around each zone's bounding box, max 100, nearest first) as a background trigger; on ENTER, run the point-in-polygon check and post a notification only if truly inside. (b) Before inference, compute the variance of the Laplacian on a 224×224 grayscale copy; if below a threshold, ask the user to retake the photo (blur lowered accuracy by ~15 points, Section 11.6)."*

Also measure on the phone: average inference time over 20 photos and peak memory (Android Studio Profiler) → give the numbers to Claude for manuscript Section 11.2's "Not yet measured" row.

## Step 7 — Optional, only if Steps 1–6 are done by 16 Oct
- A small Python script that fetches LLDA/BFAR advisory pages and inserts new text advisories into Supabase for an admin to approve.
- Filipino/Tagalog UI strings.
- A simple React admin page (otherwise the Supabase dashboard is the admin console).

## Final check before 20 Oct
- [ ] Airplane mode: identify, library, map, zone banner and log all work.
- [ ] Fresh install on the phone works (uninstall → install the demo APK).
- [ ] No keys in Git (`local.properties` not committed).
- [ ] TEST zones set `is_active = false`.
- [ ] Phone latency and memory numbers added to the manuscript.
- [ ] Backup: demo APK in Google Drive and a screen recording of every feature.
