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
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Màn hình LIÊN HỆ KHẨN CẤP — lưu người thân, đặt NGƯỠNG số lần buồn ngủ để
 * TỰ ĐỘNG gửi vị trí + thời gian qua SMS. Thiết kế lại theo theme đỏ–đen "Stitch".
 */
class EmergencyContactActivity : Activity() {

    private val PREFS = "agrypnia_prefs"
    private lateinit var nameField: EditText
    private lateinit var phoneField: EditText
    private lateinit var limitValue: TextView
    private var limit = EmergencyDispatcher.DEFAULT_LIMIT

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        limit = prefs.getInt("drowsy_limit", EmergencyDispatcher.DEFAULT_LIMIT)

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(28), dp(16), dp(28))
        }

        content.addView(UiKit.title(this, "🆘  Liên hệ khẩn cấp", UiKit.TXT, 24f).apply {
            setPadding(0, 0, 0, dp(4))
        })
        content.addView(TextView(this).apply {
            text = "Tự động gửi vị trí cho người thân khi bạn buồn ngủ quá ngưỡng an toàn."
            setTextColor(UiKit.SUB); textSize = 13f; setPadding(0, 0, 0, dp(16))
        })

        // ── Thẻ thông tin người thân ──
        val cInfo = UiKit.card(this)
        cInfo.addView(UiKit.title(this, "👤 Người thân nhận cảnh báo", UiKit.TXT, 17f))
        cInfo.addView(UiKit.label(this, "Tên người thân").apply { setPadding(0, dp(12), 0, dp(6)) })
        nameField = field(prefs.getString("contact_name", ""), "VD: Bố", false)
        cInfo.addView(nameField)
        cInfo.addView(UiKit.label(this, "Số điện thoại").apply { setPadding(0, dp(12), 0, dp(6)) })
        phoneField = field(prefs.getString("contact_phone", ""), "VD: 09xxxxxxxx", true)
        cInfo.addView(phoneField)
        cInfo.addView(Button(this).apply {
            text = "💾  Lưu liên hệ"; setTextColor(0xFFFFFFFF.toInt())
            background = UiKit.rounded(UiKit.RED, dp(14).toFloat())
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)).apply { topMargin = dp(16) }
            setOnClickListener {
                prefs.edit().putString("contact_name", nameField.text.toString())
                    .putString("contact_phone", phoneField.text.toString()).apply()
                Toast.makeText(this@EmergencyContactActivity, "Đã lưu liên hệ", Toast.LENGTH_SHORT).show()
            }
        })
        content.addView(cInfo)

        // ── Thẻ cấu hình NGƯỠNG tự động gửi ──
        val cLimit = UiKit.card(this)
        cLimit.addView(UiKit.title(this, "⚙️ Ngưỡng tự động gửi định vị", UiKit.TXT, 17f))
        cLimit.addView(TextView(this).apply {
            text = "Khi số lần buồn ngủ trong một chuyến đi đạt mức này, app sẽ TỰ ĐỘNG gửi SMS vị trí."
            setTextColor(UiKit.SUB); textSize = 13f; setPadding(0, dp(6), 0, dp(14))
        })
        val stepper = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        stepper.addView(stepBtn("−") { setLimit(limit - 1, prefs) })
        limitValue = TextView(this).apply {
            text = "$limit lần"; setTextColor(UiKit.RED); textSize = 26f; gravity = Gravity.CENTER
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        stepper.addView(limitValue)
        stepper.addView(stepBtn("+") { setLimit(limit + 1, prefs) })
        cLimit.addView(stepper)

        val autoSent = prefs.getBoolean("auto_sent_trip", false)
        val lastSentAt = prefs.getString("last_auto_sent_at", null)
        cLimit.addView(UiKit.chip(this,
            if (autoSent) "✓ Đã tự gửi lúc ${lastSentAt ?: "—"}" else "Chưa kích hoạt trong chuyến này",
            if (autoSent) UiKit.RED else UiKit.GREEN).apply {
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { topMargin = dp(14) }
        })
        content.addView(cLimit)

        // ── Thẻ thử gửi ngay ──
        val cTest = UiKit.card(this, UiKit.CARD2)
        cTest.addView(Button(this).apply {
            text = "📍  Gửi vị trí ngay (thử)"; setTextColor(0xFFFFFFFF.toInt())
            background = UiKit.roundedStroke(UiKit.CARD2, UiKit.RED, dp(14).toFloat(), dp(2))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52))
            setOnClickListener { sendLocation() }
        })
        cTest.addView(TextView(this).apply {
            text = "Nút này mở ứng dụng SMS để bạn kiểm tra nội dung. Khi vượt ngưỡng thật, " +
                   "app gửi tự động không cần thao tác."
            setTextColor(UiKit.SUB); textSize = 12f; setPadding(0, dp(10), 0, 0)
        })
        content.addView(cTest)

        val scroll = ScrollView(this).apply { setBackgroundColor(UiKit.BG); addView(content) }
        setContentView(scroll)

        // Xin quyền vị trí + SMS để tính năng tự động hoạt động
        val need = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.SEND_SMS)
            .filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (need.isNotEmpty()) ActivityCompat.requestPermissions(this, need.toTypedArray(), 1)
    }

    private fun dp(v: Int) = UiKit.dp(this, v)

    private fun field(value: String?, hintText: String, phone: Boolean) = EditText(this).apply {
        setText(value); hint = hintText
        setTextColor(0xFFFFFFFF.toInt()); setHintTextColor(UiKit.SUB)
        background = UiKit.rounded(UiKit.CARD2, dp(12).toFloat())
        setPadding(dp(14), dp(12), dp(14), dp(12))
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        if (phone) inputType = android.text.InputType.TYPE_CLASS_PHONE
    }

    private fun stepBtn(t: String, onClick: () -> Unit) = Button(this).apply {
        text = t; textSize = 22f; setTextColor(0xFFFFFFFF.toInt())
        background = UiKit.rounded(UiKit.CARD2, dp(12).toFloat())
        layoutParams = LinearLayout.LayoutParams(dp(56), dp(52))
        setOnClickListener { onClick() }
    }

    private fun setLimit(v: Int, prefs: android.content.SharedPreferences) {
        limit = v.coerceIn(1, 10)
        limitValue.text = "$limit lần"
        prefs.edit().putInt("drowsy_limit", limit).apply()
    }

    /** Lấy vị trí cuối + mở SMS soạn sẵn (link Maps + thời gian) — dùng cho nút THỬ. */
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
