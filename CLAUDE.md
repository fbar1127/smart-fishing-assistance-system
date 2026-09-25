# CLAUDE.md — rules for every coding session on IsdaYou

IsdaYou is a native Android app (Kotlin, Jetpack Compose) for municipal fishers on Laguna de Bay.
It identifies 31 fish species on-device and warns fishers about restricted zones and lake advisories,
working **fully offline**. The manuscript (Software Engineering, BSCS 3A) is the source of truth.
If code and manuscript disagree, stop and ask the user before changing either.

## Team and ownership (two builders)
- **Troy** (main PC): Android app core — `ml/`, `ui/IdentifyScreen.kt`, `geo/`, `map/`, `sync/` download, `MainActivity.kt`.
- **Diana** (laptop): data and backend — `assets/species.json`, `backend/`, `log/` (Room catch/identification log), `sync/` upload, consent screen.
- Only edit files in the current user's area. If a change is needed in the other person's files, say so instead.

## Never break these rules
1. **No pixel normalization in Kotlin.** Resize the whole photo to 224 × 224 (no crop) and pass raw RGB 0–255.
   Normalization is inside the model (manuscript Section 7.2). Input dtype is read from the model (FLOAT32 for FP16).
2. **Model:** `assets/species_classifier.tflite` = EfficientNet-Lite0 FP16 (6.5 MB). Backup: DenseNet121 FP16, same I/O.
3. **Thresholds:** primary result only if top probability ≥ **0.45**; otherwise show "Not sure" and the best guess.
   Alternatives: other classes ≥ **0.10**, at most **3**. Constants live in `FishClassifier`.
4. **labels.txt order is fixed** (0 = Bangus … 30 = Tilapia) and equals `species.json` `index` and the `species.id` in Supabase.
5. **Zone checks run offline** against polygons cached in Room, using `geo/PointInPolygon`. Internet is only for refreshing data.
6. **Android Geofencing API = trigger only** (circles, max 100 per app). The decision is always the point-in-polygon check.
7. **minSdk 28** (Android 9). Target phone for testing: Infinix Zero 30 5G (Troy's).
8. **Uploads need consent** (MLR-016). Nothing leaves the phone unless the user turned sharing on.
9. **No secrets in Git.** The Supabase URL and anon key go in `mobile-app/local.properties` (ignored), read via BuildConfig.
10. Do not invent zone boundaries, regulations or species facts. Use only data the team provides with a source.

## Stack
Kotlin 2.0, Compose Material 3, LiteRT (TensorFlow Lite) 1.0, Room (KSP), WorkManager, Play Services Location,
osmdroid (OpenStreetMap, no API key), OkHttp + kotlinx.serialization for the Supabase REST API.
Backend: Supabase (PostgreSQL + PostGIS). Schema: `backend/supabase/schema.sql`. Admins edit data in the Supabase dashboard.

## How to work
- One step from `docs/BUILD_PLAN.md` per session. Read the step first, then make the smallest change that completes it.
- After changes: build (`./gradlew :app:assembleDebug` or Android Studio ▶), run unit tests (`./gradlew :app:testDebugUnitTest`),
  and tell the user exactly what to test on the phone, including an **airplane-mode** check where it applies.
- Explain changes in plain language; the users are students learning Android.
- Commit message format: `Step N: what changed (manuscript section/requirement)`.
