package com.isdayou.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * On-device fish identification with the EfficientNet-Lite0 FP16 model (manuscript Section 10.3).
 *
 * IMPORTANT (manuscript Section 7.2): normalization is built INTO the model.
 * This class only resizes the photo to 224 × 224 and passes raw RGB values 0–255.
 * Do not divide by 255 or subtract a mean here.
 */
class FishClassifier(context: Context) : AutoCloseable {

    companion object {
        const val MODEL_FILE = "species_classifier.tflite"
        const val LABELS_FILE = "labels.txt"
        const val INPUT_SIZE = 224
        const val PRIMARY_THRESHOLD = 0.45f   // chosen on the validation set (Section 11.5)
        const val ALT_THRESHOLD = 0.10f       // MLR-007
        const val MAX_ALTERNATIVES = 3
    }

    val labels: List<String> = context.assets.open(LABELS_FILE).bufferedReader()
        .readLines().map { it.trim() }.filter { it.isNotEmpty() }

    private val interpreter: Interpreter = Interpreter(
        loadModel(context),
        Interpreter.Options().apply { setNumThreads(4) },
    )
    private val inputType: DataType = interpreter.getInputTensor(0).dataType()

    init {
        val outputs = interpreter.getOutputTensor(0).shape().last()
        check(outputs == labels.size) { "Model has $outputs outputs but $LABELS_FILE has ${labels.size} lines" }
    }

    private fun loadModel(context: Context): MappedByteBuffer {
        context.assets.openFd(MODEL_FILE).use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** Classifies a photo of one fish. Call from a background thread. */
    fun classify(photo: Bitmap): Prediction {
        val input = toInput(photo)
        val output = Array(1) { FloatArray(labels.size) }
        val start = SystemClock.elapsedRealtime()
        interpreter.run(input, output)
        val ms = SystemClock.elapsedRealtime() - start
        return Decision.decide(output[0], labels, PRIMARY_THRESHOLD, ALT_THRESHOLD, MAX_ALTERNATIVES, ms)
    }

    /** Resize the whole photo (no crop, same as training) and write raw RGB 0–255. */
    private fun toInput(photo: Bitmap): ByteBuffer {
        val argb = if (photo.config == Bitmap.Config.ARGB_8888) photo else photo.copy(Bitmap.Config.ARGB_8888, false)
        val scaled = Bitmap.createScaledBitmap(argb, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val bytesPerValue = if (inputType == DataType.FLOAT32) 4 else 1
        val buffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * bytesPerValue)
            .order(ByteOrder.nativeOrder())
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            if (inputType == DataType.FLOAT32) {
                buffer.putFloat(r.toFloat()); buffer.putFloat(g.toFloat()); buffer.putFloat(b.toFloat())
            } else { // UINT8 input (e.g. an INT8 model)
                buffer.put(r.toByte()); buffer.put(g.toByte()); buffer.put(b.toByte())
            }
        }
        buffer.rewind()
        return buffer
    }

    override fun close() = interpreter.close()
}
