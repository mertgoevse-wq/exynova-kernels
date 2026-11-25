package com.exynova.lab;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.view.MotionEvent;
import android.view.View;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 2nd ExyNova Real-Time Hardware Demo Game:
 * 128 autonomous neon drones steered by a Vulkan GPU MLP (128x8 -> 128x16 -> 128x2)
 * with optional live Samsung ENN NPU spatial hazard detection.
 */
public class NeonSwarmView extends View {
    public interface TelemetryListener {
        void onTelemetry(double gpuMs, double npuMs, double totalMs, boolean npuActive,
                         int collected, float fps);
    }

    private static final int NUM_AGENTS = 128;

    private final Paint bgPaint = new Paint();
    private final Paint gridPaint = new Paint();
    private final Paint targetPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hazardPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint agentPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint trailPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hudBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hudTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final float[] state = new float[5 + NUM_AGENTS * 4];
    private final ExecutorService aiExec = Executors.newSingleThreadExecutor();

    private volatile boolean running = false;
    private volatile boolean stepping = false;
    private volatile boolean needReset = true;
    private volatile boolean useNpuHazard = false;

    private volatile float targetX = 0.72f;
    private volatile float targetY = 0.50f;
    private volatile float hazardY = 0.50f;
    private volatile double lastGpuMs = 0.0;
    private volatile double lastNpuMs = 0.0;
    private volatile double lastTotalMs = 0.0;
    private volatile boolean lastNpuActive = false;

    private int collected = 0;
    private float goalX = 0.78f;
    private float goalY = 0.44f;
    private long seed = 1337L;
    private float liveFps = 60.0f;
    private long lastTickNs = 0L;
    private int hudTick = 0;
    private TelemetryListener listener;

    private final Runnable loopRunnable = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            stepFrame();
            invalidate();
            postDelayed(this, 16L);
        }
    };

    public NeonSwarmView(Context context) {
        super(context);
        gridPaint.setColor(Color.argb(30, 56, 189, 248));
        gridPaint.setStrokeWidth(1.5f);

        targetPaint.setStyle(Paint.Style.STROKE);
        targetPaint.setStrokeWidth(4f);
        targetPaint.setColor(Color.parseColor("#38BDF8"));

        hazardPaint.setStyle(Paint.Style.STROKE);
        hazardPaint.setStrokeWidth(3f);
        hazardPaint.setColor(Color.parseColor("#FB7185"));

        agentPaint.setStyle(Paint.Style.FILL);
        trailPaint.setStyle(Paint.Style.STROKE);
        trailPaint.setStrokeWidth(2.4f);

        hudBgPaint.setColor(Color.argb(180, 9, 13, 24));
        hudTextPaint.setTypeface(Typeface.MONOSPACE);
    }

    public void setTelemetryListener(TelemetryListener l) {
        this.listener = l;
    }

    public void setUseNpuHazard(boolean enabled) {
        this.useNpuHazard = enabled;
        invalidate();
    }

    public boolean isUseNpuHazard() {
        return useNpuHazard;
    }

    public void resetSwarm() {
        needReset = true;
        collected = 0;
        targetX = 0.72f;
        targetY = 0.50f;
        goalX = 0.78f;
        goalY = 0.42f;
        invalidate();
    }

    public void start() {
        if (running) return;
        running = true;
        lastTickNs = System.nanoTime();
        removeCallbacks(loopRunnable);
        post(loopRunnable);
    }

    public void pause() {
        running = false;
        removeCallbacks(loopRunnable);
    }

    public void release() {
        pause();
        aiExec.shutdownNow();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        bgPaint.setShader(new LinearGradient(
                0f, 0f, w, h,
                new int[]{Color.parseColor("#050814"), Color.parseColor("#0C162E"), Color.parseColor("#111D3A")},
                new float[]{0f, 0.55f, 1f},
                Shader.TileMode.CLAMP));
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        int act = event.getAction();
        if (act == MotionEvent.ACTION_DOWN || act == MotionEvent.ACTION_MOVE) {
            if (getWidth() > 0 && getHeight() > 0) {
                targetX = Math.max(0.06f, Math.min(0.94f, event.getX() / getWidth()));
                targetY = Math.max(0.16f, Math.min(0.94f, event.getY() / getHeight()));
            }
            return true;
        }
        return super.onTouchEvent(event);
    }

    private void stepFrame() {
        long now = System.nanoTime();
        if (lastTickNs > 0L) {
            double dt = (now - lastTickNs) / 1e6;
            if (dt > 1.0 && dt < 250.0) {
                liveFps = liveFps * 0.9f + (float) (1000.0 / dt) * 0.1f;
            }
        }
        lastTickNs = now;

        if (!stepping) {
            stepping = true;
            final boolean doReset = needReset;
            needReset = false;
            final boolean npuOn = useNpuHazard;
            final float tx = targetX;
            final float ty = targetY;
            aiExec.execute(() -> {
                try {
                    float[] res = ExyNovaNative.nativeStepNeonSwarm(doReset, npuOn, tx, ty);
                    if (res != null && res.length >= state.length) {
                        System.arraycopy(res, 0, state, 0, state.length);
                        lastGpuMs = res[0];
                        lastNpuMs = res[1];
                        lastTotalMs = res[2];
                        lastNpuActive = res[3] > 0.5f;
                        hazardY = res[4];
                    }
                } catch (Throwable ignored) {
                } finally {
                    stepping = false;
                }
            });
        }

        // Check goal captures across agents
        int hits = 0;
        for (int i = 0; i < NUM_AGENTS; i++) {
            int base = 5 + i * 4;
            float ax = state[base];
            float ay = state[base + 1];
            float dx = ax - goalX;
            float dy = ay - goalY;
            if (dx * dx + dy * dy < 0.0022f) {
                hits++;
            }
        }
        if (hits >= 14) {
            collected++;
            seed = seed * 1664525L + 1013904223L;
            goalX = 0.14f + ((seed & 0xFFL) / 255.0f) * 0.72f;
            seed = seed * 1664525L + 1013904223L;
            goalY = 0.22f + ((seed & 0xFFL) / 255.0f) * 0.66f;
        }

        hudTick++;
        if (listener != null && (hudTick % 6 == 0)) {
            listener.onTelemetry(lastGpuMs, lastNpuMs, lastTotalMs, lastNpuActive, collected, liveFps);
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float w = getWidth();
        float h = getHeight();
        if (w <= 0 || h <= 0) return;

        canvas.drawRect(0f, 0f, w, h, bgPaint);

        float step = Math.max(48f, w / 10f);
        for (float x = 0; x < w; x += step) {
            canvas.drawLine(x, 0f, x, h, gridPaint);
        }
        for (float y = 0; y < h; y += step) {
            canvas.drawLine(0f, y, w, y, gridPaint);
        }

        // NPU hazard line if enabled
        if (useNpuHazard) {
            float hy = Math.max(0.18f, Math.min(0.92f, hazardY)) * h;
            hazardPaint.setAlpha(180);
            canvas.drawLine(0f, hy, w, hy, hazardPaint);
        }

        // Collectible goal beacon
        float gx = goalX * w;
        float gy = goalY * h;
        agentPaint.setColor(Color.parseColor("#10B981"));
        agentPaint.setAlpha(60);
        canvas.drawCircle(gx, gy, 34f, agentPaint);
        agentPaint.setAlpha(230);
        canvas.drawCircle(gx, gy, 13f, agentPaint);

        // User finger target reticle
        float tx = targetX * w;
        float ty = targetY * h;
        canvas.drawCircle(tx, ty, 28f, targetPaint);

        // 128 GPU neural drones
        for (int i = 0; i < NUM_AGENTS; i++) {
            int base = 5 + i * 4;
            float ax = state[base] * w;
            float ay = state[base + 1] * h;
            float vx = state[base + 2] * w * 2.5f;
            float vy = state[base + 3] * h * 2.5f;
            int col = (i % 3 == 0) ? Color.parseColor("#38BDF8")
                    : (i % 3 == 1 ? Color.parseColor("#A78BFA") : Color.parseColor("#2DD4BF"));
            trailPaint.setColor(col);
            trailPaint.setAlpha(120);
            canvas.drawLine(ax, ay, ax - vx, ay - vy, trailPaint);
            agentPaint.setColor(col);
            agentPaint.setAlpha(240);
            canvas.drawCircle(ax, ay, 5.5f, agentPaint);
        }

        // Top compact HUD
        float pad = Math.max(14f, w * 0.025f);
        RectF hud = new RectF(pad, pad, w - pad, pad + Math.max(72f, h * 0.13f));
        canvas.drawRoundRect(hud, 16f, 16f, hudBgPaint);

        hudTextPaint.setColor(lastNpuActive ? Color.parseColor("#10B981") : Color.parseColor("#38BDF8"));
        hudTextPaint.setTextSize(Math.max(20f, h * 0.032f));
        canvas.drawText(String.format(Locale.US,
                "128x NEON SWARM · %s",
                lastNpuActive ? "VULKAN GPU + ENN NPU" : "VULKAN GPU"),
                hud.left + 18f, hud.top + hudTextPaint.getTextSize() + 10f, hudTextPaint);

        hudTextPaint.setColor(Color.WHITE);
        hudTextPaint.setTextSize(Math.max(19f, h * 0.030f));
        canvas.drawText(String.format(Locale.US,
                "Ziele: %d   FPS: %.0f   GPU: %.2f ms   NPU: %.2f ms",
                collected, liveFps, lastGpuMs, lastNpuMs),
                hud.left + 18f, hud.bottom - 14f, hudTextPaint);
    }
}
