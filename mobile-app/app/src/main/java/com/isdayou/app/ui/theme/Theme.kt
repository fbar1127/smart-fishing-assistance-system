package com.isdayou.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Colors = lightColorScheme(
    primary = Color(0xFF0B5E7A),      // lake blue
    secondary = Color(0xFF2E7D32),
    error = Color(0xFFB3261E),
)

@Composable
fun IsdaYouTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Colors, content = content)
}
