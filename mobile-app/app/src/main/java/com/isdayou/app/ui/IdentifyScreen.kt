package com.isdayou.app.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.isdayou.app.data.SpeciesRepository
import com.isdayou.app.ml.FishClassifier
import com.isdayou.app.ml.Prediction
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

@Composable
fun IdentifyScreen(classifier: Result<FishClassifier>, species: SpeciesRepository?) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var photo by remember { mutableStateOf<Bitmap?>(null) }
    var result by remember { mutableStateOf<Prediction?>(null) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    fun analyze(uri: Uri) {
        val model = classifier.getOrElse {
            error = "Model not loaded: ${it.message}\nCopy species_classifier.tflite and labels.txt into app/src/main/assets."
            return
        }
        busy = true; error = null; result = null
        scope.launch {
            try {
                val bmp = withContext(Dispatchers.IO) { decode(context, uri) }
                photo = bmp
                result = withContext(Dispatchers.Default) { model.classify(bmp) }
            } catch (e: Exception) {
                error = "Could not read the photo: ${e.message}"
            } finally {
                busy = false
            }
        }
    }

    val cameraUri = remember { photoUri(context) }
    val takePicture = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) analyze(cameraUri)
    }
    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) analyze(uri)
    }

    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Identify a fish", style = MaterialTheme.typography.headlineSmall)
        Text("Photograph one fish, side view, filling most of the frame.")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { takePicture.launch(cameraUri) }, enabled = !busy) { Text("Take photo") }
            OutlinedButton(
                onClick = { pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !busy,
            ) { Text("From gallery") }
        }
        photo?.let {
            Image(it.asImageBitmap(), contentDescription = "Selected photo",
                modifier = Modifier.fillMaxWidth().height(240.dp), contentScale = ContentScale.Crop)
        }
        if (busy) CircularProgressIndicator()
        error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        result?.let { ResultCard(it, species) }
    }
}

@Composable
private fun ResultCard(p: Prediction, species: SpeciesRepository?) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val primary = p.primary
            if (primary != null) {
                val info = species?.byIndex(primary.index)
                Text(primary.label, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                Text("Confidence: ${pct(primary.probability)}")
                info?.let {
                    if (it.localNames.isNotEmpty()) Text("Local names: ${it.localNames.joinToString()}")
                    if (it.scientificName.isNotBlank()) Text(it.scientificName, style = MaterialTheme.typography.bodySmall)
                    if (it.advisoryType != "none") AdvisoryBanner(it.advisoryType, it.advisoryText)
                }
            } else {
                Text("Not sure", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                Text("Best guess: ${p.top.label} (${pct(p.top.probability)}). Try a clearer photo: side view, good light, no blur.")
            }
            if (p.alternatives.isNotEmpty()) {
                Text("Other possible species:", fontWeight = FontWeight.SemiBold)
                p.alternatives.forEach { Text("• ${it.label} — ${pct(it.probability)}") }
            }
            Text("Identified on this phone in ${p.inferenceMs} ms (no internet needed)",
                style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun AdvisoryBanner(type: String, text: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
        Column(Modifier.padding(12.dp)) {
            Text("Advisory: ${type.replaceFirstChar { it.uppercase() }} species", fontWeight = FontWeight.Bold)
            if (text.isNotBlank()) Text(text)
        }
    }
}

private fun pct(x: Float) = "%.0f%%".format(x * 100)

private fun photoUri(context: Context): Uri {
    val dir = File(context.cacheDir, "photos").apply { mkdirs() }
    val file = File(dir, "capture.jpg")
    return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
}

/** Decodes with EXIF rotation applied, downsampled to at most 1024 px to save memory. */
private fun decode(context: Context, uri: Uri): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val longest = maxOf(info.size.width, info.size.height)
        if (longest > 1024) {
            val s = 1024f / longest
            decoder.setTargetSize((info.size.width * s).toInt(), (info.size.height * s).toInt())
        }
    }
}
