-- IsdaYou online database (Supabase = hosted PostgreSQL + PostGIS).
-- Run once in Supabase: Dashboard → SQL Editor → New query → paste → Run.
-- The phone app works fully offline; this database only supplies refreshed
-- reference data and receives logs the user agreed to share (Section 5.3, MLR-016).

create extension if not exists postgis;

-- 1. Species (reference data; id = line number in labels.txt)
create table if not exists species (
  id               int primary key check (id between 0 and 30),
  common_name      text not null,
  scientific_name  text default '',
  local_names      text[] default '{}',
  description      text default '',
  habitat          text default '',
  average_size     text default '',
  advisory_type    text not null default 'none' check (advisory_type in ('none','invasive')),
  advisory_text    text default '',
  source           text default '',
  updated_at       timestamptz not null default now()
);

-- 2. Restricted zones (fish sanctuaries, ZOMAP fishpen/fishcage belts, navigational lanes, ...)
create table if not exists zones (
  id           uuid primary key default gen_random_uuid(),
  name         text not null,
  zone_type    text not null check (zone_type in
                 ('fish_sanctuary','fishpen_belt','fishcage_belt','navigational_lane','other')),
  authority    text not null,              -- LLDA, BFAR, LGU ...
  rule_text    text not null,              -- what the fisher is told when inside
  boundary     geometry(MultiPolygon, 4326) not null,
  active_from  date,                        -- null = always
  active_to    date,
  source_url   text default '',
  is_active    boolean not null default true,
  updated_at   timestamptz not null default now()
);
create index if not exists zones_boundary_idx on zones using gist (boundary);

-- 3. Lake advisories (text notices, optionally tied to a zone or species)
create table if not exists advisories (
  id          uuid primary key default gen_random_uuid(),
  title       text not null,
  body        text not null,
  zone_id     uuid references zones(id) on delete set null,
  species_id  int references species(id) on delete set null,
  valid_from  date,
  valid_to    date,
  source_url  text default '',
  is_active   boolean not null default true,
  updated_at  timestamptz not null default now()
);

-- 4. Uploads from phones (only when the user consented)
create table if not exists identifications (
  id           uuid primary key,             -- generated on the phone, so re-uploads do not duplicate
  device_id    text not null,
  species_id   int references species(id),
  confidence   real not null,
  top3         jsonb not null,               -- [{"index":..,"p":..}, ...]
  confident    boolean not null,
  latitude     double precision,
  longitude    double precision,
  model_version text not null,
  created_at   timestamptz not null
);

create table if not exists catch_logs (
  id          uuid primary key,
  device_id   text not null,
  species_id  int references species(id),
  quantity    int,
  weight_kg   real,
  latitude    double precision,
  longitude   double precision,
  caught_at   timestamptz not null,
  notes       text default ''
);

-- 5. What the phone downloads: zones as GeoJSON text (easy to parse in Kotlin)
create or replace view zones_geojson as
  select id, name, zone_type, authority, rule_text, active_from, active_to, is_active, updated_at,
         st_asgeojson(boundary)::json as geometry
  from zones;

-- 6. Keep updated_at current (the phone downloads only rows changed since its last sync)
create or replace function touch_updated_at() returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end $$;
drop trigger if exists t_species on species;     create trigger t_species    before update on species    for each row execute function touch_updated_at();
drop trigger if exists t_zones on zones;         create trigger t_zones      before update on zones      for each row execute function touch_updated_at();
drop trigger if exists t_advisories on advisories; create trigger t_advisories before update on advisories for each row execute function touch_updated_at();

-- 7. Security: the app key may READ reference data and only INSERT its own logs.
--    Editing is done by admins in the Supabase dashboard (acts as the admin console).
alter table species enable row level security;
alter table zones enable row level security;
alter table advisories enable row level security;
alter table identifications enable row level security;
alter table catch_logs enable row level security;

drop policy if exists read_species on species;       create policy read_species on species for select to anon using (true);
drop policy if exists read_zones on zones;           create policy read_zones on zones for select to anon using (true);
drop policy if exists read_advisories on advisories; create policy read_advisories on advisories for select to anon using (true);
drop policy if exists add_ident on identifications;  create policy add_ident on identifications for insert to anon with check (true);
drop policy if exists add_catch on catch_logs;       create policy add_catch on catch_logs for insert to anon with check (true);
alter view zones_geojson set (security_invoker = true);
grant select on zones_geojson to anon;
