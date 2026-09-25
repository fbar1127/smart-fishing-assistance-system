package com.isdayou.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.isdayou.app.data.SpeciesRepository

/** Offline species library (BR03). Reads assets/species.json; no internet needed. */
@Composable
fun LibraryScreen(species: Result<SpeciesRepository>) {
    val repo = species.getOrElse {
        Text("Could not load species.json: ${it.message}", Modifier.padding(16.dp),
            color = MaterialTheme.colorScheme.error)
        return
    }
    var query by remember { mutableStateOf("") }
    var expanded by remember { mutableStateOf<Int?>(null) }
    val shown = repo.all.filter { s ->
        query.isBlank() || s.commonName.contains(query, true) || s.localNames.any { it.contains(query, true) }
    }
    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Species library (${repo.all.size})", style = MaterialTheme.typography.headlineSmall)
        OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth(), label = { Text("Search name or local name") })
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(shown, key = { it.index }) { s ->
                Card(Modifier.fillMaxWidth().clickable { expanded = if (expanded == s.index) null else s.index }) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(s.commonName, fontWeight = FontWeight.Bold)
                        if (s.localNames.isNotEmpty()) Text(s.localNames.joinToString())
                        if (s.advisoryType != "none") Text("Advisory: ${s.advisoryType}", color = MaterialTheme.colorScheme.error)
                        if (expanded == s.index) {
                            if (s.scientificName.isNotBlank()) Text(s.scientificName, style = MaterialTheme.typography.bodySmall)
                            if (s.description.isNotBlank()) Text(s.description)
                            if (s.habitat.isNotBlank()) Text("Habitat: ${s.habitat}")
                            if (s.averageSize.isNotBlank()) Text("Average size: ${s.averageSize}")
                            if (s.advisoryText.isNotBlank()) Text(s.advisoryText)
                        }
                    }
                }
            }
        }
    }
}
