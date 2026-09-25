"""Prints SQL that loads mobile-app/app/src/main/assets/species.json into the species table.
Usage:  python backend/supabase/seed_species.py > seed.sql   then run seed.sql in the Supabase SQL Editor.
Re-run whenever species.json changes; existing rows are updated."""
import json
from pathlib import Path

src = Path(__file__).resolve().parents[2] / "mobile-app/app/src/main/assets/species.json"
q = lambda s: "'" + str(s).replace("'", "''") + "'"
rows = []
for s in json.loads(src.read_text(encoding="utf-8")):
    names = "array[" + ",".join(q(n) for n in s["localNames"]) + "]::text[]"
    rows.append(f"({s['index']},{q(s['commonName'])},{q(s['scientificName'])},{names},{q(s['description'])},"
                f"{q(s['habitat'])},{q(s['averageSize'])},{q(s['advisoryType'])},{q(s['advisoryText'])},{q(s['source'])})")
print("insert into species (id,common_name,scientific_name,local_names,description,habitat,average_size,"
      "advisory_type,advisory_text,source) values\n" + ",\n".join(rows) +
      "\non conflict (id) do update set common_name=excluded.common_name, scientific_name=excluded.scientific_name,"
      " local_names=excluded.local_names, description=excluded.description, habitat=excluded.habitat,"
      " average_size=excluded.average_size, advisory_type=excluded.advisory_type,"
      " advisory_text=excluded.advisory_text, source=excluded.source;")
