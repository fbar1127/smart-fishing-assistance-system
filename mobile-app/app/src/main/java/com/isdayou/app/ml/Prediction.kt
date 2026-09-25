package com.isdayou.app.ml

/** One class and the probability the model gave it. */
data class Candidate(val index: Int, val label: String, val probability: Float)

/**
 * The result shown to the user.
 * - [primary] is set only when the top probability reaches [FishClassifier.PRIMARY_THRESHOLD]
 *   (manuscript MLR-006/MLR-008: withhold a species name when confidence is low).
 * - [alternatives] are the other classes at or above [FishClassifier.ALT_THRESHOLD], at most 3 (MLR-007).
 */
data class Prediction(
    val top: Candidate,
    val primary: Candidate?,
    val alternatives: List<Candidate>,
    val inferenceMs: Long,
) {
    val isConfident: Boolean get() = primary != null
}

/** Pure decision logic, kept separate from the TFLite code so it can be unit-tested. */
object Decision {
    fun decide(
        probs: FloatArray,
        labels: List<String>,
        primaryThreshold: Float,
        altThreshold: Float,
        maxAlternatives: Int,
        inferenceMs: Long = 0,
    ): Prediction {
        require(probs.size == labels.size) { "Model has ${probs.size} outputs but labels.txt has ${labels.size} lines" }
        val ranked = probs.indices.sortedByDescending { probs[it] }
            .map { Candidate(it, labels[it], probs[it]) }
        val top = ranked.first()
        val primary = if (top.probability >= primaryThreshold) top else null
        val alternatives = ranked.drop(1)
            .filter { it.probability >= altThreshold }
            .take(maxAlternatives)
        return Prediction(top, primary, alternatives, inferenceMs)
    }
}
