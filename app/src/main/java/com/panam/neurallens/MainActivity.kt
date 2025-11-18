package com.panam.neurallens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var imageEnhancer: ImageEnhancer
    private lateinit var imageView: ImageView
    private lateinit var tvLabel: TextView
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button
    private lateinit var btnEnhance: Button
    private lateinit var btnCompare: Button
    private lateinit var rgMode: RadioGroup
    private var originalBitmap: Bitmap? = null
    private var enhancedBitmap: Bitmap? = null
    private var showingOriginal = true
    private lateinit var photoFile: File

    companion object {
        private const val MAX_IMAGE_DIMENSION = 1920
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            originalBitmap = decodeSampledBitmapFromFile(photoFile.absolutePath)
            imageView.setImageBitmap(originalBitmap)
            btnEnhance.isEnabled = true
            btnCompare.isEnabled = false
            enhancedBitmap?.recycle()
            enhancedBitmap = null
            tvLabel.text = "Original"
            showingOriginal = true
        }
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            try {
                val inputStream = contentResolver.openInputStream(it)
                val loadedBitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (loadedBitmap != null) {
                    // Resize if too large
                    originalBitmap?.recycle()
                    originalBitmap = resizeBitmapIfNeeded(loadedBitmap)

                    imageView.setImageBitmap(originalBitmap)
                    btnEnhance.isEnabled = true
                    btnCompare.isEnabled = false
                    enhancedBitmap?.recycle()
                    enhancedBitmap = null
                    tvLabel.text = "Original (from Gallery)"
                    showingOriginal = true
                } else {
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            }
        }
    }

    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            launchCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageEnhancer = ImageEnhancer(this)
        imageView = findViewById(R.id.imageView)
        tvLabel = findViewById(R.id.tvLabel)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        btnEnhance = findViewById(R.id.btnEnhance)
        btnCompare = findViewById(R.id.btnCompare)
        rgMode = findViewById(R.id.rgMode)

        photoFile = File(cacheDir, "photo.jpg")

        btnCamera.setOnClickListener {
            checkCameraPermission()
        }

        btnGallery.setOnClickListener {
            pickImage.launch("image/*")
        }

        btnEnhance.setOnClickListener {
            enhanceImage()
        }

        btnCompare.setOnClickListener {
            toggleComparison()
        }
    }

    private fun decodeSampledBitmapFromFile(path: String): Bitmap {
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(path, options)

        val (width, height) = options.outWidth to options.outHeight
        var inSampleSize = 1

        if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
            val halfWidth = width / 2
            val halfHeight = height / 2

            while (halfWidth / inSampleSize >= MAX_IMAGE_DIMENSION &&
                halfHeight / inSampleSize >= MAX_IMAGE_DIMENSION) {
                inSampleSize *= 2
            }
        }

        return BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
        }.let { BitmapFactory.decodeFile(path, it) }
    }

    private fun resizeBitmapIfNeeded(bitmap: Bitmap): Bitmap {
        val maxDim = MAX_IMAGE_DIMENSION
        return if (bitmap.width > maxDim || bitmap.height > maxDim) {
            val scale = maxDim.toFloat() / maxOf(bitmap.width, bitmap.height)
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true).also {
                bitmap.recycle()
            }
        } else {
            bitmap
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                launchCamera()
            }
            else -> {
                requestPermission.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun launchCamera() {
        val uri = FileProvider.getUriForFile(this, "$packageName.provider", photoFile)
        takePicture.launch(uri)
    }

    private fun enhanceImage() {
        val bitmap = originalBitmap ?: return

        val mode = when(rgMode.checkedRadioButtonId) {
            R.id.rbLowLight -> EnhancementMode.LOW_LIGHT
            R.id.rbSharpen -> EnhancementMode.SHARPEN
            R.id.rbDeblur -> EnhancementMode.DEBLUR
            R.id.rbBoth -> EnhancementMode.BOTH
            else -> EnhancementMode.BOTH
        }

        Toast.makeText(this, "Processing with ${mode.name}...", Toast.LENGTH_SHORT).show()
        btnEnhance.isEnabled = false

        lifecycleScope.launch {
            val enhanced = withContext(Dispatchers.Default) {
                imageEnhancer.enhance(bitmap, mode)
            }
            enhancedBitmap?.recycle()
            enhancedBitmap = enhanced
            imageView.setImageBitmap(enhanced)
            tvLabel.text = "Enhanced (${mode.name})"
            showingOriginal = false
            btnEnhance.isEnabled = true
            btnCompare.isEnabled = true
            Toast.makeText(this@MainActivity, "Done! Click Compare to toggle", Toast.LENGTH_SHORT).show()
        }
    }

    private fun toggleComparison() {
        if (showingOriginal) {
            imageView.setImageBitmap(enhancedBitmap)
            tvLabel.text = "Enhanced"
        } else {
            imageView.setImageBitmap(originalBitmap)
            tvLabel.text = "Original"
        }
        showingOriginal = !showingOriginal
    }

    override fun onDestroy() {
        super.onDestroy()
        imageEnhancer.release()
        originalBitmap?.recycle()
        enhancedBitmap?.recycle()
    }
}