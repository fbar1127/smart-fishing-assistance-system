package com.isdayou.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.isdayou.app.data.SpeciesRepository
import com.isdayou.app.ml.FishClassifier
import com.isdayou.app.ui.IdentifyScreen
import com.isdayou.app.ui.LibraryScreen
import com.isdayou.app.ui.theme.IsdaYouTheme

class MainActivity : ComponentActivity() {

    // Loaded once. If a file is missing, the error is shown on screen instead of crashing.
    private val classifier: Result<FishClassifier> by lazy { runCatching { FishClassifier(this) } }
    private val species: Result<SpeciesRepository> by lazy { runCatching { SpeciesRepository(this) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            IsdaYouTheme { App(classifier, species) }
        }
    }

    override fun onDestroy() {
        classifier.getOrNull()?.close()
        super.onDestroy()
    }
}

private data class Tab(val title: String, val icon: ImageVector)

private val tabs = listOf(
    Tab("Identify", Icons.Filled.Search),
    Tab("Library", Icons.Filled.List),
    Tab("Map", Icons.Filled.Place),
    Tab("Log", Icons.Filled.Edit),
)

@Composable
private fun App(classifier: Result<FishClassifier>, species: Result<SpeciesRepository>) {
    var selected by rememberSaveable { mutableIntStateOf(0) }
    Scaffold(
        bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { i, tab ->
                    NavigationBarItem(
                        selected = selected == i,
                        onClick = { selected = i },
                        icon = { Icon(tab.icon, contentDescription = tab.title) },
                        label = { Text(tab.title) },
                    )
                }
            }
        }
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when (selected) {
                0 -> IdentifyScreen(classifier, species.getOrNull())
                1 -> LibraryScreen(species)
                2 -> ComingSoon("Map and restricted-zone check — Step 3")
                else -> ComingSoon("Catch and identification log — Step 4")
            }
        }
    }
}

@Composable
private fun ComingSoon(text: String) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) { Text(text) }
}
