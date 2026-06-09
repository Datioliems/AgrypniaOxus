package com.ai2026.drowsydriver

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import android.view.animation.AnimationUtils
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * SplashActivity — màn hình khởi động AgrypniaOxus
 *
 * Luồng animation:
 *  1. Logo fade-in + scale từ 0.7 → 1.0  (600ms, delay 200ms)
 *  2. Dừng ở 1.0 trong 900ms
 *  3. Logo phóng to + fade-out: scale 1.0 → 2.2  (550ms)
 *  4. Chuyển sang MainActivity (không animation — cảm giác "lao vào" màn hình chính)
 *
 * Tổng thời gian splash ≈ 2.3s
 */
class SplashActivity : AppCompatActivity() {

    private val HOLD_MS = 900L          // thời gian logo dừng ở trạng thái bình thường
    private val ANIM_IN_MS = 800L       // khớp tổng thời gian splash_logo_in.xml  (200+600)
    private val ANIM_OUT_MS = 550L      // khớp splash_logo_out.xml

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Toàn màn hình — ẩn status bar + nav bar
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
            android.view.View.SYSTEM_UI_FLAG_FULLSCREEN
                or android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            )

        setContentView(R.layout.activity_splash)

        val logo   = findViewById<ImageView>(R.id.splash_logo)
        val title  = findViewById<TextView>(R.id.splash_title)

        val animIn  = AnimationUtils.loadAnimation(this, R.anim.splash_logo_in)
        val animOut = AnimationUtils.loadAnimation(this, R.anim.splash_logo_out)

        // Phase 1: logo xuất hiện
        logo.startAnimation(animIn)
        title.alpha = 0f
        title.animate().alpha(0.7f).setStartDelay(400).setDuration(400).start()

        // Phase 2: dừng → Phase 3: phóng to mờ
        Handler(Looper.getMainLooper()).postDelayed({
            logo.startAnimation(animOut)
            title.animate().alpha(0f).setDuration(ANIM_OUT_MS).start()

            // Phase 4: vào MainActivity ngay khi animation out kết thúc
            Handler(Looper.getMainLooper()).postDelayed({
                startActivity(Intent(this, MainActivity::class.java))
                // Không overridePendingTransition — tạo cảm giác "lao thẳng" vào app
                overridePendingTransition(android.R.anim.fade_in, 0)
                finish()
            }, ANIM_OUT_MS)

        }, ANIM_IN_MS + HOLD_MS)
    }
}
