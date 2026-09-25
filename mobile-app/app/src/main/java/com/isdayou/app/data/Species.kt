package com.isdayou.app.data

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * One entry of assets/species.json (the offline species library, BR03).
 * [index] must equal the line number (0-based) of the species in labels.txt.
 */
@Serializable
data class Species(
    val index: Int,
    val commonName: String,
    val scientificName: String = "",
    val localNames: List<String> = emptyList(),
    val description: String = "",
    val habitat: String = "",
    val averageSize: String = "",
    val advisoryType: String = "none",   // "none" or "invasive" (Knifefish, Janitor Fish)
    val advisoryText: String = "",
    val source: String = "",
)

class SpeciesRepository(context: Context) {
    private val json = Json { ignoreUnknownKeys = true }

    val all: List<Species> = json.decodeFromString<List<Species>>(
        context.assets.open("species.json").bufferedReader().readText()
    ).sortedBy { it.index }

    fun byIndex(index: Int): Species? = all.firstOrNull { it.index == index }
}
