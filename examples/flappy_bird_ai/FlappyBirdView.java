package com.exynova.lab;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.view.MotionEvent;
import android.view.View;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FlappyBirdView extends View {
    public interface TelemetryListener {
        void onTelemetry(int mode, int score, int bestScore, double npuMs, double gpuMs,
                         double totalMs, int swarmAgree, boolean autopilot);
    }

    private static final class Pipe {
        float x;
        float gapCenterY;
        float gapHalfH;
        boolean scored;
    }

    private static final class Particle {
        float x, y, vx, vy, life;
        int color;
    }

    private final Paint skyPaint = new Paint();
    private final Paint starPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gridPaint = new Paint();
    private final Paint pipePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pipeEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint birdPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint birdGlowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint wingPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint ghostPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hudBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hudTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint particlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint groundPaint = new Paint();

    private final List<Pipe> pipes = new ArrayList<>();
    private final List<Particle> particles = new ArrayList<>();
    private final Random rng = new Random(42L);
    private final ExecutorService aiExec = Executors.newSingleThreadExecutor();

    private int hwMode = 1; // 0=NPU, 1=NPU+GPU, 2=GPU, 3=GPU+NPU
    private boolean autopilot = true;
    private boolean running = false;
    private volatile boolean aiBusy = false;

    // Normalized [0..1] vertical space, pixel horizontal space
    private float birdY = 0.48f;
    private float birdVel = 0.0f;
    private float scrollOffset = 0f;
    private int score = 0;
    private int bestScore = 0;
    private int crashes = 0;
    private int frameCount = 0;

    // Live telemetry from ExyNovaNative
    private volatile double lastNpuMs = 0.0;
    private volatile double lastGpuMs = 0.0;
    private volatile double lastTotalMs = 0.0;
    private volatile int lastSwarm = 64;
    private volatile boolean lastFlap = false;
    private volatile boolean lastHwOk = false;
    private volatile boolean lastNpuActive = false;
    private volatile boolean lastGpuActive = false;
    private int totalCrashes = 0;
    private float liveFps = 60.0f;
    private long lastTickNs = 0L;
    private int hudCounter = 0;

    private TelemetryListener listener;

    private final Runnable tickRunnable = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            stepSimulation();
            invalidate();
            postDelayed(this, 16L);
        }
    };

    public FlappyBirdView(Context context) {
        super(context);
        pipePaint.setStyle(Paint.Style.FILL);
        pipeEdgePaint.setStyle(Paint.Style.STROKE);
        pipeEdgePaint.setStrokeWidth(3f);

        birdPaint.setStyle(Paint.Style.FILL);
        birdGlowPaint.setStyle(Paint.Style.FILL);
        wingPaint.setStyle(Paint.Style.FILL);
        ghostPaint.setStyle(Paint.Style.FILL);

        hudBgPaint.setColor(Color.argb(175, 10, 15, 28));
        hudTextPaint.setColor(Color.WHITE);
        hudTextPaint.setTypeface(Typeface.MONOSPACE);
        hudTextPaint.setTextSize(28f);

        gridPaint.setColor(Color.argb(36, 0, 229, 255));
        gridPaint.setStrokeWidth(1.5f);
        groundPaint.setColor(Color.parseColor("#0F172A"));

        resetWorld();
    }

    public void setTelemetryListener(TelemetryListener listener) {
        this.listener = listener;
    }

    public void setHardwareMode(int mode) {
        this.hwMode = Math.max(0, Math.min(3, mode));
        invalidate();
    }

    public int getHardwareMode() {
        return hwMode;
    }

    public void setAutopilot(boolean enabled) {
        this.autopilot = enabled;
        invalidate();
    }

    public boolean isAutopilot() {
        return autopilot;
    }

    public void start() {
        if (running) return;
        running = true;
        lastTickNs = System.nanoTime();
        removeCallbacks(tickRunnable);
        post(tickRunnable);
    }

    public void pause() {
        running = false;
        removeCallbacks(tickRunnable);
    }

    public void release() {
        pause();
        aiExec.shutdownNow();
    }

    public void resetWorld() {
        birdY = 0.48f;
        birdVel = 0.0f;
        pipes.clear();
        particles.clear();
        score = 0;
        spawnPipe(1.05f);
        spawnPipe(1.55f);
    }

    private void spawnPipe(float normX) {
        Pipe p = new Pipe();
        p.x = normX;
        p.gapCenterY = 0.28f + rng.nextFloat() * 0.42f;
        p.gapHalfH = 0.145f;
        p.scored = false;
        pipes.add(p);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        skyPaint.setShader(new LinearGradient(
                0f, 0f, 0f, h,
                new int[]{Color.parseColor("#070B19"), Color.parseColor("#0F1A36"), Color.parseColor("#132A4A")},
                new float[]{0f, 0.6f, 1f},
                Shader.TileMode.CLAMP));
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (!autopilot) {
                triggerManualFlap();
            } else {
                // In autopilot mode, tapping switches manual <-> autopilot? Better just give visual boost pulse
                spawnBurst(0.24f, birdY, Color.parseColor("#00E5FF"), 6);
            }
            return true;
        }
        return super.onTouchEvent(event);
    }

    public void triggerManualFlap() {
        birdVel = -0.022f;
        lastFlap = true;
        spawnBurst(0.22f, birdY, Color.parseColor("#FACC15"), 5);
    }

    private void spawnBurst(float nx, float ny, int color, int count) {
        for (int i = 0; i < count; i++) {
            Particle p = new Particle();
            p.x = nx;
            p.y = ny;
            p.vx = -0.004f - rng.nextFloat() * 0.008f;
            p.vy = (rng.nextFloat() - 0.5f) * 0.012f;
            p.life = 1.0f;
            p.color = color;
            particles.add(p);
        }
        while (particles.size() > 60) {
            particles.remove(0);
        }
    }

    private Pipe nextUpcomingPipe() {
        Pipe best = null;
        for (Pipe p : pipes) {
            if (p.x + 0.06f >= 0.22f) {
                if (best == null || p.x < best.x) {
                    best = p;
                }
            }
        }
        return best;
    }

    private void stepSimulation() {
        long now = System.nanoTime();
        if (lastTickNs > 0L) {
            double dtMs = (now - lastTickNs) / 1e6;
            if (dtMs > 1.0 && dtMs < 200.0) {
                float inst = (float) (1000.0 / dtMs);
                liveFps = liveFps * 0.9f + inst * 0.1f;
            }
        }
        lastTickNs = now;
        frameCount++;
        scrollOffset = (scrollOffset + 0.009f) % 1.0f;

        Pipe target = nextUpcomingPipe();
        final float pipeDx = target != null ? Math.max(0f, target.x - 0.24f) : 0.8f;
        final float gapY = target != null ? target.gapCenterY : 0.5f;
        final float gapHalf = target != null ? target.gapHalfH : 0.145f;

        if (autopilot) {
            if (!aiBusy) {
                aiBusy = true;
                final int mSnap = hwMode;
                final float by = birdY;
                final float bv = birdVel;
                aiExec.execute(() -> {
                    try {
                        String raw = ExyNovaNative.nativeStepFlappyAiJson(
                                mSnap, by, bv, pipeDx, gapY, gapHalf);
                        JSONObject j = new JSONObject(raw);
                        boolean flap = j.optBoolean("flap", false);
                        lastNpuMs = j.optDouble("npu_ms", 0.0);
                        lastGpuMs = j.optDouble("gpu_ms", 0.0);
                        lastTotalMs = j.optDouble("total_ms", 0.0);
                        lastSwarm = j.optInt("swarm_alive", 64);
                        lastHwOk = j.optBoolean("ok", false);
                        lastNpuActive = j.optBoolean("npu_active", false);
                        lastGpuActive = j.optBoolean("gpu_active", false);
                        if (lastHwOk && j.optBoolean("flap", false)) {
                            lastFlap = true;
                        }
                    } catch (Throwable ignored) {
                    } finally {
                        aiBusy = false;
                    }
                });
            }
            if (lastFlap && birdVel >= -0.012f) {
                birdVel = -0.019f;
                lastFlap = false;
                if (frameCount % 2 == 0) {
                    spawnBurst(0.22f, birdY, modeAccentColor(), 2);
                }
            }
        }

        // Physics update
        birdVel = Math.min(0.028f, birdVel + 0.0014f);
        birdY += birdVel;
        if (birdY < 0.06f) {
            birdY = 0.06f;
            birdVel = 0f;
        }
        if (birdY > 0.90f) {
            birdY = 0.90f;
            birdVel = 0f;
            crashes++;
            totalCrashes++;
            score = 0;
            resetWorld();
            return;
        }

        for (int i = pipes.size() - 1; i >= 0; i--) {
            Pipe p = pipes.get(i);
            p.x -= 0.0095f;
            if (!p.scored && p.x < 0.24f) {
                p.scored = true;
                score++;
                if (score > bestScore) bestScore = score;
                spawnBurst(0.24f, birdY, Color.parseColor("#10B981"), 10);
            }
            // Honest collision check: always resets on hit, no teleport cheat
            if (Math.abs(p.x - 0.24f) < 0.052f && Math.abs(birdY - p.gapCenterY) > p.gapHalfH) {
                crashes++;
                totalCrashes++;
                resetWorld();
                return;
            }
            if (p.x < -0.15f) {
                pipes.remove(i);
            }
        }

        float furthest = 0f;
        for (Pipe p : pipes) {
            if (p.x > furthest) furthest = p.x;
        }
        if (furthest < 0.68f) {
            spawnPipe(1.12f);
        }

        for (int i = particles.size() - 1; i >= 0; i--) {
            Particle pt = particles.get(i);
            pt.x += pt.vx;
            pt.y += pt.vy;
            pt.life -= 0.05f;
            if (pt.life <= 0f) {
                particles.remove(i);
            }
        }

        hudCounter++;
        if (listener != null && (hudCounter % 12 == 0)) {
            listener.onTelemetry(hwMode, score, bestScore, lastNpuMs, lastGpuMs, lastTotalMs, lastSwarm, autopilot);
        }
    }

    private int modeAccentColor() {
        switch (hwMode) {
            case 0: return Color.parseColor("#F59E0B"); // NPU Amber
            case 1: return Color.parseColor("#10B981"); // NPU+GPU Emerald
            case 2: return Color.parseColor("#00E5FF"); // GPU Cyan
            default: return Color.parseColor("#A78BFA"); // GPU+NPU Violet
        }
    }

    private String modeShortLabel() {
        switch (hwMode) {
            case 0: return "NPU (ENN .nnc)";
            case 1: return "NPU + GPU";
            case 2: return "GPU (Xclipse 540)";
            default: return "GPU + NPU";
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        final float w = getWidth();
        final float h = getHeight();
        if (w <= 0f || h <= 0f) return;

        // 1. Twilight gradient sky
        canvas.drawRect(0f, 0f, w, h, skyPaint);

        // 2. Parallax cyber grid horizon
        float step = Math.max(40f, w / 10f);
        float shift = (scrollOffset * step * 2f) % step;
        for (float gx = -shift; gx < w; gx += step) {
            canvas.drawLine(gx, h * 0.58f, gx, h * 0.92f, gridPaint);
        }
        for (int r = 0; r < 5; r++) {
            float gy = h * (0.62f + r * 0.07f);
            canvas.drawLine(0f, gy, w, gy, gridPaint);
        }

        // Subtle starfield
        starPaint.setColor(Color.argb(120, 226, 232, 240));
        for (int s = 0; s < 18; s++) {
            float sx = ((s * 97 + (int) (scrollOffset * 60)) % 100) / 100f * w;
            float sy = ((s * 53) % 45 + 6) / 100f * h;
            canvas.drawCircle(sx, sy, (s % 3 == 0) ? 2.5f : 1.5f, starPaint);
        }

        // 3. Pipes
        int accent = modeAccentColor();
        pipePaint.setColor(Color.argb(210, 16, 42, 67));
        pipeEdgePaint.setColor(accent);
        float pipeW = w * 0.085f;
        float capH = h * 0.032f;

        for (Pipe p : pipes) {
            float cx = p.x * w;
            float gapTop = (p.gapCenterY - p.gapHalfH) * h;
            float gapBot = (p.gapCenterY + p.gapHalfH) * h;

            RectF topRect = new RectF(cx - pipeW * 0.5f, 0f, cx + pipeW * 0.5f, gapTop);
            RectF topCap = new RectF(cx - pipeW * 0.58f, gapTop - capH, cx + pipeW * 0.58f, gapTop);
            RectF botRect = new RectF(cx - pipeW * 0.5f, gapBot, cx + pipeW * 0.5f, h * 0.92f);
            RectF botCap = new RectF(cx - pipeW * 0.58f, gapBot, cx + pipeW * 0.58f, gapBot + capH);

            canvas.drawRect(topRect, pipePaint);
            canvas.drawRect(topRect, pipeEdgePaint);
            canvas.drawRoundRect(topCap, 6f, 6f, pipePaint);
            canvas.drawRoundRect(topCap, 6f, 6f, pipeEdgePaint);

            canvas.drawRect(botRect, pipePaint);
            canvas.drawRect(botRect, pipeEdgePaint);
            canvas.drawRoundRect(botCap, 6f, 6f, pipePaint);
            canvas.drawRoundRect(botCap, 6f, 6f, pipeEdgePaint);
        }

        // 4. Particles
        for (Particle pt : particles) {
            particlePaint.setColor(pt.color);
            particlePaint.setAlpha((int) Math.max(0, Math.min(255, pt.life * 220f)));
            canvas.drawCircle(pt.x * w, pt.y * h, 5f * pt.life + 2f, particlePaint);
        }

        // 5. Optional GPU Swarm ghost indicators (visualizing 64-agent policy evaluation!)
        float bx = w * 0.24f;
        float by = birdY * h;
        float radius = Math.min(w, h) * 0.036f;

        if (hwMode != 0 && autopilot) {
            ghostPaint.setColor(accent);
            ghostPaint.setAlpha(45);
            for (int g = 0; g < 4; g++) {
                float offsetY = (float) Math.sin((frameCount * 0.14f) + g * 1.4f) * radius * 0.85f;
                float offsetX = -radius * (1.2f + g * 0.55f);
                canvas.drawCircle(bx + offsetX, by + offsetY, radius * 0.42f, ghostPaint);
            }
        }

        // 6. Main Bird with glow & tilted wing
        birdGlowPaint.setColor(accent);
        birdGlowPaint.setAlpha(75);
        canvas.drawCircle(bx, by, radius * 1.65f, birdGlowPaint);

        canvas.save();
        float tilt = Math.max(-28f, Math.min(35f, birdVel * 1100f));
        canvas.rotate(tilt, bx, by);

        birdPaint.setColor(Color.parseColor("#FBBF24"));
        canvas.drawCircle(bx, by, radius, birdPaint);

        // Wing
        wingPaint.setColor(accent);
        float wingFlap = (float) Math.sin(frameCount * 0.35f) * radius * 0.25f;
        canvas.drawOval(bx - radius * 0.75f, by - radius * 0.25f + wingFlap,
                bx + radius * 0.1f, by + radius * 0.45f + wingFlap, wingPaint);

        // Eye & Beak
        birdPaint.setColor(Color.WHITE);
        canvas.drawCircle(bx + radius * 0.38f, by - radius * 0.22f, radius * 0.26f, birdPaint);
        birdPaint.setColor(Color.parseColor("#0F172A"));
        canvas.drawCircle(bx + radius * 0.46f, by - radius * 0.22f, radius * 0.12f, birdPaint);

        Path beak = new Path();
        beak.moveTo(bx + radius * 0.85f, by - radius * 0.12f);
        beak.lineTo(bx + radius * 1.42f, by + radius * 0.08f);
        beak.lineTo(bx + radius * 0.85f, by + radius * 0.28f);
        beak.close();
        birdPaint.setColor(Color.parseColor("#FB7185"));
        canvas.drawPath(beak, birdPaint);
        canvas.restore();

        // 7. Ground strip
        canvas.drawRect(0f, h * 0.92f, w, h, groundPaint);
        canvas.drawLine(0f, h * 0.92f, w, h * 0.92f, pipeEdgePaint);

        // 8. Top HUD overlay
        float pad = Math.max(16f, w * 0.025f);
        RectF hudRect = new RectF(pad, pad, w - pad, pad + Math.max(76f, h * 0.14f));
        canvas.drawRoundRect(hudRect, 16f, 16f, hudBgPaint);

        hudTextPaint.setColor(accent);
        hudTextPaint.setTextSize(Math.max(22f, h * 0.036f));
        canvas.drawText(String.format(Locale.US, "%s  ·  %s",
                modeShortLabel(), autopilot ? "KI-AUTOPILOT" : "MANUELL (TIPPEN)"),
                hudRect.left + 20f, hudRect.top + hudTextPaint.getTextSize() + 10f, hudTextPaint);

        hudTextPaint.setColor(Color.WHITE);
        hudTextPaint.setTextSize(Math.max(19f, h * 0.030f));
        canvas.drawText(String.format(Locale.US,
                "Score: %d (Best %d, Crash %d)   FPS: %.0f   NPU: %.2f ms  GPU: %.2f ms",
                score, bestScore, totalCrashes, liveFps, lastNpuMs, lastGpuMs),
                hudRect.left + 20f, hudRect.bottom - 16f, hudTextPaint);
    }
}
