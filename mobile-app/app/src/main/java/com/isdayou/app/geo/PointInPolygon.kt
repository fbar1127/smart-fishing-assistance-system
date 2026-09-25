package com.isdayou.app.geo

/** A point in WGS84 degrees. */
data class LatLng(val lat: Double, val lng: Double)

/**
 * Offline restricted-zone check (manuscript Section 5.3, checklist Step 7.6).
 * The Android Geofencing API only supports circles (max 100), so it is used only as a
 * background trigger; this ray-casting test on the cached polygon makes the real decision.
 */
object PointInPolygon {

    /** [ring] is the outer boundary; the closing point may be repeated or not. */
    fun contains(ring: List<LatLng>, p: LatLng): Boolean {
        if (ring.size < 3) return false
        var inside = false
        var j = ring.size - 1
        for (i in ring.indices) {
            val a = ring[i]
            val b = ring[j]
            val crosses = (a.lat > p.lat) != (b.lat > p.lat) &&
                p.lng < (b.lng - a.lng) * (p.lat - a.lat) / (b.lat - a.lat) + a.lng
            if (crosses) inside = !inside
            j = i
        }
        return inside
    }

    /** Polygon with optional holes (GeoJSON order: first ring outer, rest holes). */
    fun containsWithHoles(rings: List<List<LatLng>>, p: LatLng): Boolean {
        if (rings.isEmpty() || !contains(rings[0], p)) return false
        return rings.drop(1).none { contains(it, p) }
    }
}
