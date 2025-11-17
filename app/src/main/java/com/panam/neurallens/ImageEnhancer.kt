package com.panam.neurallens

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import java.nio.FloatBuffer

class ImageEnhancer(context: Context) {
    private val session: OrtSession
    private val env = OrtEnvironment.getEnvironment()

    init {
        Log.d("ImageEnhancer", "Loading Real-ESRGAN model...")
        val modelBytes = context.assets.open("realesrgan.onnx").readBytes()
        session = env.createSession(modelBytes)
        Log.d("ImageEnhancer", "Model loaded! Input name: ${session.inputNames.first()}")
    }

    fun enhance(bitmap: Bitmap): Bitmap {
        val startTime = System.currentTimeMillis()

        // Real-ESRGAN expects 256x256 input
        val inputSize = 256
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Convert to float array [1, 3, 256, 256], normalized to [0, 1]
        val floatArray = FloatArray(1 * 3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Convert to CHW format (Channel, Height, Width)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            floatArray[i] = r
            floatArray[inputSize * inputSize + i] = g
            floatArray[2 * inputSize * inputSize + i] = b
        }

        Log.d("ImageEnhancer", "Input prepared, running model...")

        // Run inference
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArray),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val results = session.run(mapOf(session.inputNames.first() to inputTensor))
        val outputTensor = results[0].value as Array<Array<Array<FloatArray>>>

        Log.d("ImageEnhancer", "Model output shape: [${outputTensor.size}, ${outputTensor[0].size}, ${outputTensor[0][0].size}, ${outputTensor[0][0][0].size}]")

        // Output is 4x larger (1024x1024)
        val outputSize = inputSize * 4
        val enhanced = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
        val outputPixels = IntArray(outputSize * outputSize)

        for (i in 0 until outputSize * outputSize) {
            val y = i / outputSize
            val x = i % outputSize

            val r = (outputTensor[0][0][y][x].coerceIn(0f, 1f) * 255).toInt()
            val g = (outputTensor[0][1][y][x].coerceIn(0f, 1f) * 255).toInt()
            val b = (outputTensor[0][2][y][x].coerceIn(0f, 1f) * 255).toInt()

            outputPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        enhanced.setPixels(outputPixels, 0, outputSize, 0, 0, outputSize, outputSize)

        val duration = System.currentTimeMillis() - startTime
        Log.d("ImageEnhancer", "Done in ${duration}ms! Upscaled from ${inputSize}x${inputSize} to ${outputSize}x${outputSize}")

        // Return enhanced image (scaled down to 2x original size for display)
        return Bitmap.createScaledBitmap(enhanced, bitmap.width * 2, bitmap.height * 2, true)
    }
}