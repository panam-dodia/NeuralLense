package com.panam.neurallens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
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
    private lateinit var btnEnhance: Button
    private lateinit var btnCompare: Button
    private var originalBitmap: Bitmap? = null
    private var enhancedBitmap: Bitmap? = null
    private var showingOriginal = true
    private lateinit var photoFile: File

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            originalBitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
            imageView.setImageBitmap(originalBitmap)
            btnEnhance.isEnabled = true
            btnCompare.isEnabled = false
            enhancedBitmap = null
            tvLabel.text = "Original"
            showingOriginal = true
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
        btnEnhance = findViewById(R.id.btnEnhance)
        btnCompare = findViewById(R.id.btnCompare)

        photoFile = File(cacheDir, "photo.jpg")

        btnCamera.setOnClickListener {
            checkCameraPermission()
        }

        btnEnhance.setOnClickListener {
            enhanceImage()
        }

        btnCompare.setOnClickListener {
            toggleComparison()
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

        Toast.makeText(this, "Processing...", Toast.LENGTH_SHORT).show()
        btnEnhance.isEnabled = false

        lifecycleScope.launch {
            val enhanced = withContext(Dispatchers.Default) {
                imageEnhancer.enhance(bitmap)
            }
            enhancedBitmap = enhanced
            imageView.setImageBitmap(enhanced)
            tvLabel.text = "Enhanced"
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
}