package com.isdayou.app.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class DecisionTest {
    private val labels = listOf("A", "B", "C", "D", "E")

    @Test fun confidentResultWithAlternatives() {
        val p = Decision.decide(floatArrayOf(0.60f, 0.20f, 0.12f, 0.05f, 0.03f), labels, 0.45f, 0.10f, 3)
        assertEquals("A", p.primary?.label)
        assertEquals(listOf("B", "C"), p.alternatives.map { it.label })
    }

    @Test fun lowConfidenceWithholdsName() {
        val p = Decision.decide(floatArrayOf(0.40f, 0.35f, 0.15f, 0.06f, 0.04f), labels, 0.45f, 0.10f, 3)
        assertNull(p.primary)
        assertEquals("A", p.top.label)
    }

    @Test fun atMostThreeAlternatives() {
        val p = Decision.decide(floatArrayOf(0.46f, 0.14f, 0.14f, 0.13f, 0.13f), labels, 0.45f, 0.10f, 3)
        assertEquals(3, p.alternatives.size)
    }
}
