# IsdaYou — Smart Fishing Assistance System

An offline-first Android app for municipal fishers on Laguna de Bay: on-device fish identification (31 species,
EfficientNet-Lite0, TensorFlow Lite), an offline species library, GPS-based restricted-zone and lake-advisory
warnings, and catch logging with optional sync.

BSCS 3A · Software Engineering · University of Cebu Lapu-Lapu Mandaue
Team: Baring, Francis Troy T. · Sotto, Franc Louwi · Ugbaniel, Jake · Villaraza, Diana

| Folder | Contents |
|---|---|
| `mobile-app/` | Android Studio project (Kotlin, Jetpack Compose, minSdk 28) |
| `backend/supabase/` | Database schema (PostgreSQL + PostGIS on Supabase), zone template, species seeder |
| `ml/` | Dataset audit and training scripts (run in Colab/Kaggle); model details |
| `docs/BUILD_PLAN.md` | Two-person build plan to 20 Oct 2026, with a Claude prompt for each step |
| `CLAUDE.md` | Rules every coding session must follow |

## Run the app
1. Open `mobile-app/` in Android Studio and let Gradle sync.
2. Copy `species_classifier.tflite` into `mobile-app/app/src/main/assets/` (see the README there).
3. Connect the phone with USB debugging on, and press ▶ Run.
