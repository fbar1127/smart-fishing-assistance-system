-- How to add one restricted zone (Diana):
-- 1. Open https://geojson.io, zoom to the zone on Laguna de Bay, draw it with the polygon tool,
--    following the official map (LLDA ZOMAP, BFAR or LGU ordinance).
-- 2. Copy ONLY the "geometry" object from the right panel: {"type":"Polygon","coordinates":[...]}
-- 3. Paste it below, fill in the other fields, and run in Supabase SQL Editor.
-- Only encode zones from an official source, and put that source in source_url.

insert into zones (name, zone_type, authority, rule_text, source_url, boundary)
values (
  'NAME OF ZONE',
  'fish_sanctuary',          -- fish_sanctuary | fishpen_belt | fishcage_belt | navigational_lane | other
  'LLDA',
  'Fishing is not allowed inside this fish sanctuary.',
  'https://...official source...',
  st_multi(st_setsrid(st_geomfromgeojson('PASTE GEOMETRY HERE'), 4326))
);

-- For testing only: draw a small zone around your own house or school, name it 'TEST — ...',
-- and set is_active = false before the final demo.
