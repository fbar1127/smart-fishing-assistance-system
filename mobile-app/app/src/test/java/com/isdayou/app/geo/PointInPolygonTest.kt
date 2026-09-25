package com.isdayou.app.geo

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PointInPolygonTest {
    // A made-up square for testing only (not a real zone).
    private val square = listOf(
        LatLng(14.20, 121.10), LatLng(14.20, 121.20), LatLng(14.30, 121.20), LatLng(14.30, 121.10),
    )

    @Test fun insideIsDetected() = assertTrue(PointInPolygon.contains(square, LatLng(14.25, 121.15)))
    @Test fun outsideIsRejected() = assertFalse(PointInPolygon.contains(square, LatLng(14.35, 121.15)))
    @Test fun holeIsExcluded() {
        val hole = listOf(LatLng(14.24, 121.14), LatLng(14.24, 121.16), LatLng(14.26, 121.16), LatLng(14.26, 121.14))
        assertFalse(PointInPolygon.containsWithHoles(listOf(square, hole), LatLng(14.25, 121.15)))
        assertTrue(PointInPolygon.containsWithHoles(listOf(square, hole), LatLng(14.21, 121.11)))
    }
}
