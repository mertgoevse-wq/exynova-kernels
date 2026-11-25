# ExyNova Flappy Bird AI — 4 Hardware Acceleration Modes

An AI-accelerated Flappy Bird gameplay + physics simulation example built with the ExyNova SDK.

## 4 Hardware Modes Benchmarked

1. **`NPU`**: Runs Samsung ENN `.nnc` inference via `libenn_public_api_cpp.so` (`/dev/vertex10`) per frame for obstacle/state perception.
2. **`NPU + GPU`**: Stage 1 runs ENN `.nnc` perception on the NPU; Stage 2 feeds those activations into a 64-agent Vulkan compute MLP policy (`matmul` + `relu`) on Xclipse 540.
3. **`GPU`**: Pure Xclipse 540 Vulkan compute (`conv2d_nchw` 16x16 obstacle grid + 64-agent policy MLP `matmul` + `relu`).
4. **`GPU + NPU`**: Vulkan GPU computes the primary 64-agent swarm action first, followed by an ENN NPU hazard verification pass.

## Build & Run

Host simulation & CTest:
```bash
cmake -S . -B build-host -DEXYNOVA_BUILD_HOST_TOOLS=ON -DEXYNOVA_BUILD_ANDROID=OFF
cmake --build build-host --config Release -j 4
ctest --test-dir build-host -R flappy_bird_ai_bench --output-on-failure
```

Android APK: Play interactively or watch the AI Autopilot live in the **Flappy AI** tab of `ExyNova Lab`.
