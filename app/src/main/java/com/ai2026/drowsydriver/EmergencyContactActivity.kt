package com.ai2026.drowsydriver

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Màn hình LIÊN HỆ NGƯỜI THÂN — lưu tên/SĐT, gửi VỊ TRÍ + THỜI GIAN gần nhất qua SMS
 * khi tài xế buồn ngủ. Vị trí kèm link Google Maps.
 */
class EmergencyContactActivity : Activity() {

    private val PREFS = "agrypnia_prefs"
    private lateinit var nameField: EditText
    private lateinit var phoneField: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 48)
            setBackgroundColor(0xFF131313.toInt())
        }
        fun label(t: String) = TextView(this).apply {
            text = t; setTextColor(0xFFE5E2E1.toInt()); textSize = 15f
            setPadding(0, 24, 0, 8)
        }
        root.addView(TextView(this).apply {
            text = "🆘 Liên hệ người thân"; setTextColor(0xFFFC1C46.toInt())
            textSize = 26f; setPadding(0, 0, 0, 16)
        })
        root.addView(label("Tên người thân"))
        nameField = EditText(this).apply {
            setText(prefs.getString("contact_name", "")); setTextColor(0xFFFFFFFF.toInt()); hint = "VD: Bố"
        }
        root.addView(nameField)
        root.addView(label("Số điện thoại"))
        phoneField = EditText(this).apply {
            setText(prefs.getString("contact_phone", "")); setTextColor(0xFFFFFFFF.toInt())
            hint = "VD: 09xxxxxxxx"; inputType = android.text.InputType.TYPE_CLASS_PHONE
        }
        root.addView(phoneField)

        root.addView(Button(this).apply {
            text = "💾 Lưu liên hệ"; setBackgroundColor(0xFFFC1C46.toInt()); setTextColor(0xFFFFFFFF.toInt())
            val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 140); lp.topMargin = 32; layoutParams = lp
            setOnClickListener {
                prefs.edit().putString("contact_name", nameField.text.toString())
                    .putString("contact_phone", phoneField.text.toString()).apply()
                Toast.makeText(this@EmergencyContactActivity, "Đã lưu", Toast.LENGTH_SHORT).show()
            }
        })
        root.addView(Button(this).apply {
            text = "📍 Gửi vị trí ngay (thử)"; setBackgroundColor(0xFF2A2A2A.toInt()); setTextColor(0xFFFFFFFF.toInt())
            val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 140); lp.topMargin = 16; layoutParams = lp
            setOnClickListener { sendLocation() }
        })
        root.addView(TextView(this).apply {
            text = "\nKhi phát hiện buồn ngủ kéo dài, app sẽ tự gợi ý gửi vị trí + thời gian cho người thân."
            setTextColor(0xFFAAAAAA.toInt()); textSize = 13f; setPadding(0, 24, 0, 0)
        })
        setContentView(root)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 1)
        }
    }

    /** Lấy vị trí cuối + mở SMS soạn sẵn (link Maps + thời gian). */
    fun sendLocation() {
        val phone = phoneField.text.toString().ifBlank {
            getSharedPreferences(PREFS, MODE_PRIVATE).getString("contact_phone", "") ?: ""
        }
        if (phone.isBlank()) { Toast.makeText(this, "Chưa có SĐT người thân", Toast.LENGTH_SHORT).show(); return }
        val lm = getSystemService(LOCATION_SERVICE) as LocationManager
        val loc = try {
            lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
        } catch (e: SecurityException) { null }
        val time = SimpleDateFormat("HH:mm dd/MM", Locale.getDefault()).format(Date())
        val place = if (loc != null) "https://maps.google.com/?q=${loc.latitude},${loc.longitude}" else "(chưa lấy được GPS)"
        val msg = "AgrypniaOxus: Tài xế có dấu hiệu buồn ngủ lúc $time. Vị trí: $place"
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$phone")).apply { putExtra("sms_body", msg) }
        try { startActivity(intent) } catch (e: Exception) {
            Toast.makeText(this, "Không mở được SMS", Toast.LENGTH_SHORT).show()
        }
    }
}
