#include "exynova/enn_backend.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef EXYNOVA_ANDROID
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace exynova {
namespace npu {

// Exact ABI structs reverse-engineered from Galaxy A56 libenn_public_api_cpp.so (ENN_VERSION_2.3.30)
namespace abi {

using EnnModelId = uint64_t;

enum enn_buf_dir_e : uint32_t {
    ENN_DIR_IN = 0,
    ENN_DIR_OUT = 1,
    ENN_DIR_EXT = 2,
    ENN_DIR_NONE = 3,
    ENN_DIR_SIZE = 4
};

struct _ennBuffer {
    void* va;
    uint32_t size;
    uint32_t offset;
};

struct _NumberOfBuffersInfo {
    uint32_t n_in_buf;
    uint32_t n_out_buf;
};

// sizeof(_ennBufferInfo) == 40 bytes on arm64-v8a
struct _ennBufferInfo {
    bool is_able_to_update;
    uint32_t n;
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint32_t size;
    uint32_t buffer_type;
    const char* label;
};

static_assert(sizeof(_ennBuffer) == 16, "_ennBuffer size mismatch");
static_assert(sizeof(_NumberOfBuffersInfo) == 8, "_NumberOfBuffersInfo size mismatch");
static_assert(sizeof(_ennBufferInfo) == 40, "_ennBufferInfo size mismatch");

// Exact Itanium-mangled symbol names exported by /vendor/lib64/libenn_public_api_cpp.so
constexpr const char* kSymVerStr = "_ZN3enn7ver_strE";
constexpr const char* kSymInitialize = "_ZN3enn3api13EnnInitializeEv";
constexpr const char* kSymDeinitialize = "_ZN3enn3api15EnnDeinitializeEv";
constexpr const char* kSymOpenModel = "_ZN3enn3api12EnnOpenModelEPKcPm";
constexpr const char* kSymOpenModelFromMemory = "_ZN3enn3api22EnnOpenModelFromMemoryEPKcjPm";
constexpr const char* kSymCloseModel = "_ZN3enn3api13EnnCloseModelEm";
constexpr const char* kSymGetBuffersInfo = "_ZN3enn3api17EnnGetBuffersInfoEP20_NumberOfBuffersInfom";
constexpr const char* kSymGetBufferInfoByIndex = "_ZN3enn3api23EnnGetBufferInfoByIndexEP14_ennBufferInfom14_enn_buf_dir_ej";
constexpr const char* kSymAllocateAllBuffers = "_ZN3enn3api21EnnAllocateAllBuffersEmPPP10_ennBufferP20_NumberOfBuffersInfoib";
constexpr const char* kSymReleaseBuffers = "_ZN3enn3api17EnnReleaseBuffersEPP10_ennBufferi";
constexpr const char* kSymExecuteModel = "_ZN3enn3api15EnnExecuteModelEmi";
constexpr const char* kSymSetPrefPerfMode = "_ZN3enn3api24EnnSetPreferencePerfModeEj";
constexpr const char* kSymSetPrefPresetId = "_ZN3enn3api24EnnSetPreferencePresetIdEj";
constexpr const char* kSymGetMetaInfo = "_ZN3enn3api14EnnGetMetaInfoE19_enn_meta_type_id_emPc";

using FnInitialize = int32_t (*)();
using FnDeinitialize = int32_t (*)();
using FnOpenModel = int32_t (*)(const char* path, EnnModelId* model_id);
using FnOpenModelFromMemory = int32_t (*)(const char* data, uint32_t size, EnnModelId* model_id);
using FnCloseModel = int32_t (*)(EnnModelId model_id);
using FnGetBuffersInfo = int32_t (*)(_NumberOfBuffersInfo* info, EnnModelId model_id);
using FnGetBufferInfoByIndex = int32_t (*)(_ennBufferInfo* out, EnnModelId model_id, enn_buf_dir_e dir, uint32_t idx);
using FnAllocateAllBuffers = int32_t (*)(EnnModelId model_id, _ennBuffer*** out_bufs, _NumberOfBuffersInfo* out_info, int session_id, bool do_commit);
using FnReleaseBuffers = int32_t (*)(_ennBuffer** bufs, int num_bufs);
using FnExecuteModel = int32_t (*)(EnnModelId model_id, int session_id);
using FnSetPrefPerfMode = int32_t (*)(uint32_t val);
using FnSetPrefPresetId = int32_t (*)(uint32_t val);
using FnGetMetaInfo = int32_t (*)(uint32_t meta_id, EnnModelId model_id, char* out);

} // namespace abi

struct EnnNpuBackend::EnnSymbols {
    const char** ver_str = nullptr;
    abi::FnInitialize EnnInitialize = nullptr;
    abi::FnDeinitialize EnnDeinitialize = nullptr;
    abi::FnOpenModel EnnOpenModel = nullptr;
    abi::FnOpenModelFromMemory EnnOpenModelFromMemory = nullptr;
    abi::FnCloseModel EnnCloseModel = nullptr;
    abi::FnGetBuffersInfo EnnGetBuffersInfo = nullptr;
    abi::FnGetBufferInfoByIndex EnnGetBufferInfoByIndex = nullptr;
    abi::FnAllocateAllBuffers EnnAllocateAllBuffers = nullptr;
    abi::FnReleaseBuffers EnnReleaseBuffers = nullptr;
    abi::FnExecuteModel EnnExecuteModel = nullptr;
    abi::FnSetPrefPerfMode EnnSetPreferencePerfMode = nullptr;
    abi::FnSetPrefPresetId EnnSetPreferencePresetId = nullptr;
    abi::FnGetMetaInfo EnnGetMetaInfo = nullptr;
};

struct EnnNpuBackend::OpenSessionState {
    bool open = false;
    abi::EnnModelId model_id = 0;
    abi::_ennBuffer** bufs = nullptr;
    abi::_NumberOfBuffersInfo alloc_info{};
    size_t in0_size = 0;
    size_t out0_size = 0;
};

namespace {

uint64_t fnv1a_64(const uint8_t* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

const char* nnapi_device_type_name(int32_t t) {
    switch (t) {
    case 1: return "OTHER";
    case 2: return "CPU";
    case 3: return "GPU";
    case 4: return "ACCELERATOR";
    default: return "UNKNOWN";
    }
}

} // namespace

NnapiProbeReport probe_nnapi() {
    NnapiProbeReport report;
#ifndef EXYNOVA_ANDROID
    report.error_msg = "NNAPI probe requires Android target";
    return report;
#else
    const char* candidates[] = {
        "libneuralnetworks.so",
        "/apex/com.android.neuralnetworks/lib64/libneuralnetworks.so",
        "/system/lib64/libneuralnetworks.so"
    };
    void* handle = nullptr;
    std::string errs;
    for (const char* path : candidates) {
        dlerror();
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            report.library_loaded = true;
            report.library_path = path;
            break;
        }
        const char* e = dlerror();
        errs += std::string(path) + ": " + (e ? e : "unknown") + "; ";
    }
    if (!handle) {
        report.error_msg = errs;
        return report;
    }

    using FnGetCount = int (*)(uint32_t*);
    using FnGetDev = int (*)(uint32_t, void**);
    using FnGetName = int (*)(const void*, const char**);
    using FnGetType = int (*)(const void*, int32_t*);
    using FnGetVer = int (*)(const void*, const char**);

    auto get_count = reinterpret_cast<FnGetCount>(dlsym(handle, "ANeuralNetworks_getDeviceCount"));
    auto get_dev = reinterpret_cast<FnGetDev>(dlsym(handle, "ANeuralNetworks_getDevice"));
    auto get_name = reinterpret_cast<FnGetName>(dlsym(handle, "ANeuralNetworksDevice_getName"));
    auto get_type = reinterpret_cast<FnGetType>(dlsym(handle, "ANeuralNetworksDevice_getType"));
    auto get_ver = reinterpret_cast<FnGetVer>(dlsym(handle, "ANeuralNetworksDevice_getVersion"));

    if (!get_count || !get_dev || !get_name || !get_type) {
        report.error_msg = "NNAPI symbols missing in " + report.library_path;
        dlclose(handle);
        return report;
    }

    uint32_t count = 0;
    int rc = get_count(&count);
    if (rc != 0) {
        report.error_msg = "ANeuralNetworks_getDeviceCount failed rc=" + std::to_string(rc);
        dlclose(handle);
        return report;
    }

    for (uint32_t i = 0; i < count; ++i) {
        void* dev = nullptr;
        if (get_dev(i, &dev) != 0 || !dev) continue;
        NnapiDeviceDesc desc;
        const char* nm = nullptr;
        int32_t tp = 0;
        const char* vr = nullptr;
        if (get_name(dev, &nm) == 0 && nm) desc.name = nm;
        if (get_type(dev, &tp) == 0) {
            desc.type = tp;
            desc.type_name = nnapi_device_type_name(tp);
        }
        if (get_ver && get_ver(dev, &vr) == 0 && vr) desc.version = vr;
        report.devices.push_back(std::move(desc));
    }
    dlclose(handle);
    return report;
#endif
}

EnnNpuBackend::EnnNpuBackend()
    : syms_(new EnnSymbols()), session_(new OpenSessionState()) {}

EnnNpuBackend::~EnnNpuBackend() {
    shutdown();
    delete session_;
    session_ = nullptr;
    delete syms_;
    syms_ = nullptr;
}

EnnProbeReport EnnNpuBackend::probe_and_initialize(bool set_perf_mode) {
    if (initialized_) {
        return probe_report_;
    }

    probe_report_ = EnnProbeReport{};
#ifndef EXYNOVA_ANDROID
    probe_report_.status = runtime::CapabilityStatus::UNSUPPORTED;
    probe_report_.error_msg = "ENN NPU backend requires Android ARM64 device";
    return probe_report_;
#else
    probe_report_.vertex10_exists = (access("/dev/vertex10", F_OK) == 0);
    probe_report_.vertex10_rw = (access("/dev/vertex10", R_OK | W_OK) == 0);
    if (probe_report_.vertex10_exists) {
        probe_report_.status = runtime::CapabilityStatus::DETECTED;
    }

    const char* candidates[] = {
        "/vendor/lib64/libenn_public_api_cpp.so",
        "libenn_public_api_cpp.so"
    };
    std::string load_errs;
    for (const char* path : candidates) {
        dlerror();
        lib_handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (lib_handle_) {
            probe_report_.library_loaded = true;
            probe_report_.library_path = path;
            break;
        }
        const char* e = dlerror();
        load_errs += std::string(path) + ": " + (e ? e : "unknown") + "; ";
    }

    if (!lib_handle_) {
        probe_report_.status = probe_report_.vertex10_exists
            ? runtime::CapabilityStatus::ACCESS_DENIED
            : runtime::CapabilityStatus::UNSUPPORTED;
        probe_report_.error_msg = "dlopen failed: " + load_errs;
        return probe_report_;
    }

    probe_report_.status = runtime::CapabilityStatus::ACCESSIBLE;

    auto resolve = [&](const char* sym) -> void* {
        dlerror();
        return dlsym(lib_handle_, sym);
    };

    syms_->ver_str = reinterpret_cast<const char**>(resolve(abi::kSymVerStr));
    syms_->EnnInitialize = reinterpret_cast<abi::FnInitialize>(resolve(abi::kSymInitialize));
    syms_->EnnDeinitialize = reinterpret_cast<abi::FnDeinitialize>(resolve(abi::kSymDeinitialize));
    syms_->EnnOpenModel = reinterpret_cast<abi::FnOpenModel>(resolve(abi::kSymOpenModel));
    syms_->EnnOpenModelFromMemory = reinterpret_cast<abi::FnOpenModelFromMemory>(resolve(abi::kSymOpenModelFromMemory));
    syms_->EnnCloseModel = reinterpret_cast<abi::FnCloseModel>(resolve(abi::kSymCloseModel));
    syms_->EnnGetBuffersInfo = reinterpret_cast<abi::FnGetBuffersInfo>(resolve(abi::kSymGetBuffersInfo));
    syms_->EnnGetBufferInfoByIndex = reinterpret_cast<abi::FnGetBufferInfoByIndex>(resolve(abi::kSymGetBufferInfoByIndex));
    syms_->EnnAllocateAllBuffers = reinterpret_cast<abi::FnAllocateAllBuffers>(resolve(abi::kSymAllocateAllBuffers));
    syms_->EnnReleaseBuffers = reinterpret_cast<abi::FnReleaseBuffers>(resolve(abi::kSymReleaseBuffers));
    syms_->EnnExecuteModel = reinterpret_cast<abi::FnExecuteModel>(resolve(abi::kSymExecuteModel));
    syms_->EnnSetPreferencePerfMode = reinterpret_cast<abi::FnSetPrefPerfMode>(resolve(abi::kSymSetPrefPerfMode));
    syms_->EnnSetPreferencePresetId = reinterpret_cast<abi::FnSetPrefPresetId>(resolve(abi::kSymSetPrefPresetId));
    syms_->EnnGetMetaInfo = reinterpret_cast<abi::FnGetMetaInfo>(resolve(abi::kSymGetMetaInfo));

    if (syms_->ver_str && *syms_->ver_str) {
        probe_report_.enn_version = *syms_->ver_str;
    }

    if (!syms_->EnnInitialize || !syms_->EnnDeinitialize || !syms_->EnnOpenModel ||
        !syms_->EnnCloseModel || !syms_->EnnGetBuffersInfo ||
        !syms_->EnnAllocateAllBuffers || !syms_->EnnReleaseBuffers ||
        !syms_->EnnExecuteModel) {
        probe_report_.status = runtime::CapabilityStatus::UNSUPPORTED;
        probe_report_.error_msg = "One or more required mangled enn::api symbols missing";
        return probe_report_;
    }
    probe_report_.symbols_resolved = true;

    // Ensure NDK Binder threadpool is running (safe/idempotent inside Android Apps,
    // required in standalone CLI binaries; never shrink an already-started pool).
    if (void* binder_lib = dlopen("libbinder_ndk.so", RTLD_NOW | RTLD_LOCAL)) {
        using FnStartPool = void (*)();
        auto start_pool = reinterpret_cast<FnStartPool>(dlsym(binder_lib, "ABinderProcess_startThreadPool"));
        if (start_pool) start_pool();
    }

    const int32_t rc = syms_->EnnInitialize();
    probe_report_.init_ret_code = rc;
    if (rc != static_cast<int32_t>(EnnStatus::SUCCESS)) {
        probe_report_.status = (rc == static_cast<int32_t>(EnnStatus::SERVICE_NULL))
            ? runtime::CapabilityStatus::ACCESS_DENIED
            : runtime::CapabilityStatus::EXPERIMENTAL;
        probe_report_.error_msg = std::string("EnnInitialize returned ") +
            enn_status_to_string(rc) + " (" + std::to_string(rc) + ")";
        return probe_report_;
    }

    initialized_ = true;
    probe_report_.initialized = true;
    probe_report_.status = runtime::CapabilityStatus::INITIALIZED;

    if (syms_->EnnGetMetaInfo) {
        char buf[256] = {};
        if (syms_->EnnGetMetaInfo(0x70, 0, buf) == 0) {
            probe_report_.meta_version_302 = buf;
        }
    }
    if (set_perf_mode && syms_->EnnSetPreferencePerfMode) {
        syms_->EnnSetPreferencePerfMode(1);
    }

    return probe_report_;
#endif
}

void EnnNpuBackend::shutdown() {
    close_session();
#ifdef EXYNOVA_ANDROID
    if (initialized_ && syms_ && syms_->EnnDeinitialize) {
        syms_->EnnDeinitialize();
    }
    initialized_ = false;
    if (lib_handle_) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
    }
#else
    initialized_ = false;
#endif
}

bool EnnNpuBackend::has_open_session() const {
    return session_ && session_->open;
}

size_t EnnNpuBackend::session_input_size() const {
    return session_ ? session_->in0_size : 0u;
}

size_t EnnNpuBackend::session_output_size() const {
    return session_ ? session_->out0_size : 0u;
}

void EnnNpuBackend::close_session() {
#ifdef EXYNOVA_ANDROID
    if (!session_ || !session_->open) {
        return;
    }
    if (syms_ && syms_->EnnReleaseBuffers && session_->bufs) {
        const int total = static_cast<int>(session_->alloc_info.n_in_buf + session_->alloc_info.n_out_buf);
        syms_->EnnReleaseBuffers(session_->bufs, total);
    }
    if (syms_ && syms_->EnnCloseModel && session_->model_id != 0) {
        syms_->EnnCloseModel(session_->model_id);
    }
    session_->open = false;
    session_->model_id = 0;
    session_->bufs = nullptr;
    session_->alloc_info = abi::_NumberOfBuffersInfo{};
    session_->in0_size = 0;
    session_->out0_size = 0;
#else
    if (session_) {
        session_->open = false;
    }
#endif
}

bool EnnNpuBackend::open_session(const std::string& nnc_path, std::string* err,
                                 uint32_t perf_preset_id) {
#ifndef EXYNOVA_ANDROID
    (void)nnc_path;
    (void)perf_preset_id;
    if (err) *err = "ENN persistent session requires Android device";
    return false;
#else
    close_session();
    if (!initialized_ && !probe_and_initialize(false).initialized) {
        if (err) *err = probe_report_.error_msg.empty() ? "EnnInitialize failed" : probe_report_.error_msg;
        return false;
    }
    if (perf_preset_id > 0 && syms_->EnnSetPreferencePresetId) {
        syms_->EnnSetPreferencePresetId(perf_preset_id);
    }

    abi::EnnModelId model_id = 0;
    int32_t rc = syms_->EnnOpenModel(nnc_path.c_str(), &model_id);
    if (rc != 0 && syms_->EnnOpenModelFromMemory) {
        std::ifstream ifs(nnc_path, std::ios::binary | std::ios::ate);
        if (ifs.is_open()) {
            const std::streamsize sz = ifs.tellg();
            if (sz > 0) {
                ifs.seekg(0, std::ios::beg);
                std::vector<char> mem_buf(static_cast<size_t>(sz));
                if (ifs.read(mem_buf.data(), sz)) {
                    rc = syms_->EnnOpenModelFromMemory(
                        mem_buf.data(), static_cast<uint32_t>(mem_buf.size()), &model_id);
                }
            }
        }
    }
    if (rc != 0) {
        if (err) *err = "EnnOpenModel failed: " + std::string(enn_status_to_string(rc)) + " (" + std::to_string(rc) + ")";
        return false;
    }

    abi::_ennBuffer** bufs = nullptr;
    abi::_NumberOfBuffersInfo alloc_info{};
    rc = syms_->EnnAllocateAllBuffers(model_id, &bufs, &alloc_info, 0, true);
    if (rc != 0 || !bufs) {
        syms_->EnnCloseModel(model_id);
        if (err) *err = "EnnAllocateAllBuffers failed: " + std::string(enn_status_to_string(rc)) + " (" + std::to_string(rc) + ")";
        return false;
    }

    session_->open = true;
    session_->model_id = model_id;
    session_->bufs = bufs;
    session_->alloc_info = alloc_info;
    session_->in0_size = (alloc_info.n_in_buf > 0 && bufs[0]) ? bufs[0]->size : 0u;
    session_->out0_size = (alloc_info.n_out_buf > 0 && bufs[alloc_info.n_in_buf])
        ? bufs[alloc_info.n_in_buf]->size : 0u;
    return true;
#endif
}

bool EnnNpuBackend::execute_session(const void* in_bytes, size_t in_len,
                                    std::vector<uint8_t>* out_bytes,
                                    double* exec_ms,
                                    std::string* err) {
#ifndef EXYNOVA_ANDROID
    (void)in_bytes;
    (void)in_len;
    (void)out_bytes;
    (void)exec_ms;
    if (err) *err = "ENN persistent session requires Android device";
    return false;
#else
    if (!session_ || !session_->open || !session_->bufs) {
        if (err) *err = "No open ENN session";
        return false;
    }
    const auto& info = session_->alloc_info;
    if (info.n_in_buf > 0 && session_->bufs[0] && session_->bufs[0]->va) {
        uint8_t* dst = static_cast<uint8_t*>(session_->bufs[0]->va);
        const size_t dst_sz = session_->bufs[0]->size;
        if (in_bytes && in_len > 0) {
            const size_t copy_n = (in_len < dst_sz) ? in_len : dst_sz;
            std::memcpy(dst, in_bytes, copy_n);
            if (copy_n < dst_sz) {
                std::memset(dst + copy_n, 0, dst_sz - copy_n);
            }
        } else {
            for (size_t b = 0; b < dst_sz; ++b) {
                dst[b] = static_cast<uint8_t>((b % 13u) + 1u);
            }
        }
    }
    for (uint32_t i = info.n_in_buf; i < info.n_in_buf + info.n_out_buf; ++i) {
        if (session_->bufs[i] && session_->bufs[i]->va && session_->bufs[i]->size > 0) {
            std::memset(session_->bufs[i]->va, 0, session_->bufs[i]->size);
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const int32_t rc = syms_->EnnExecuteModel(session_->model_id, 0);
    const auto t1 = std::chrono::steady_clock::now();
    if (exec_ms) {
        *exec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    if (rc != 0) {
        if (err) *err = "EnnExecuteModel failed: " + std::string(enn_status_to_string(rc)) + " (" + std::to_string(rc) + ")";
        return false;
    }
    if (out_bytes && info.n_out_buf > 0 && session_->bufs[info.n_in_buf] &&
        session_->bufs[info.n_in_buf]->va) {
        const abi::_ennBuffer* ob = session_->bufs[info.n_in_buf];
        const uint8_t* src = static_cast<const uint8_t*>(ob->va);
        out_bytes->assign(src, src + ob->size);
    }
    probe_report_.status = runtime::CapabilityStatus::EXECUTION_VERIFIED;
    return true;
#endif
}

EnnModelRunReport EnnNpuBackend::inspect_and_run_model(const std::string& nnc_path,
                                                       bool execute_inference,
                                                       uint32_t perf_preset_id) {
    EnnModelRunReport rep;
#ifndef EXYNOVA_ANDROID
    (void)nnc_path;
    (void)execute_inference;
    (void)perf_preset_id;
    rep.error_msg = "ENN NPU model execution requires Android device";
    return rep;
#else
    if (!initialized_) {
        rep.error_msg = "EnnNpuBackend is not initialized";
        return rep;
    }

    if (perf_preset_id > 0 && syms_->EnnSetPreferencePresetId) {
        syms_->EnnSetPreferencePresetId(perf_preset_id);
    }

    const auto t0 = std::chrono::steady_clock::now();
    abi::EnnModelId model_id = 0;
    int32_t rc = syms_->EnnOpenModel(nnc_path.c_str(), &model_id);
    if (rc != 0 && syms_->EnnOpenModelFromMemory) {
        std::ifstream ifs(nnc_path, std::ios::binary | std::ios::ate);
        if (ifs.is_open()) {
            const std::streamsize sz = ifs.tellg();
            if (sz > 0) {
                ifs.seekg(0, std::ios::beg);
                std::vector<char> mem_buf(static_cast<size_t>(sz));
                if (ifs.read(mem_buf.data(), sz)) {
                    rc = syms_->EnnOpenModelFromMemory(
                        mem_buf.data(), static_cast<uint32_t>(mem_buf.size()), &model_id);
                }
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    rep.open_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rep.last_ret_code = rc;
    rep.last_ret_name = enn_status_to_string(rc);
    if (rc != 0) {
        rep.error_msg = "EnnOpenModel(" + nnc_path + ") failed: " +
                        rep.last_ret_name + " (" + std::to_string(rc) + ")";
        return rep;
    }

    rep.opened = true;
    rep.model_id = model_id;

    abi::_NumberOfBuffersInfo num_info{};
    rc = syms_->EnnGetBuffersInfo(&num_info, model_id);
    rep.last_ret_code = rc;
    rep.last_ret_name = enn_status_to_string(rc);
    if (rc != 0) {
        rep.error_msg = "EnnGetBuffersInfo failed: " + rep.last_ret_name +
                        " (" + std::to_string(rc) + ")";
        syms_->EnnCloseModel(model_id);
        return rep;
    }

    auto query_dir = [&](abi::enn_buf_dir_e dir, uint32_t count, bool is_out,
                         std::vector<EnnTensorDesc>& out_vec) {
        for (uint32_t i = 0; i < count; ++i) {
            EnnTensorDesc td;
            td.index = i;
            td.is_output = is_out;
            if (syms_->EnnGetBufferInfoByIndex) {
                abi::_ennBufferInfo bi{};
                if (syms_->EnnGetBufferInfoByIndex(&bi, model_id, dir, i) == 0) {
                    td.n = bi.n;
                    td.width = bi.width;
                    td.height = bi.height;
                    td.channel = bi.channel;
                    td.size_bytes = bi.size;
                    td.buffer_type = bi.buffer_type;
                    if (bi.label) td.label = bi.label;
                }
            }
            out_vec.push_back(std::move(td));
        }
    };

    query_dir(abi::ENN_DIR_IN, num_info.n_in_buf, false, rep.inputs);
    query_dir(abi::ENN_DIR_OUT, num_info.n_out_buf, true, rep.outputs);

    if (execute_inference) {
        abi::_ennBuffer** bufs = nullptr;
        abi::_NumberOfBuffersInfo alloc_info{};
        // Verified A56 signature: 5 args (model_id, out_bufs, out_info, session_id=0, do_commit=true)
        rc = syms_->EnnAllocateAllBuffers(model_id, &bufs, &alloc_info, 0, true);
        rep.last_ret_code = rc;
        rep.last_ret_name = enn_status_to_string(rc);
        if (rc != 0 || !bufs) {
            rep.error_msg = "EnnAllocateAllBuffers failed: " + rep.last_ret_name +
                            " (" + std::to_string(rc) + ")";
            syms_->EnnCloseModel(model_id);
            return rep;
        }
        rep.buffers_allocated = true;

        const uint32_t total_bufs = alloc_info.n_in_buf + alloc_info.n_out_buf;
        for (uint32_t i = 0; i < alloc_info.n_in_buf; ++i) {
            if (bufs[i] && bufs[i]->va && bufs[i]->size > 0) {
                // Deterministic non-zero pattern (safe across FP32/INT8/INT16 inputs)
                uint8_t* ptr = static_cast<uint8_t*>(bufs[i]->va);
                for (uint32_t b = 0; b < bufs[i]->size; ++b) {
                    ptr[b] = static_cast<uint8_t>((b % 13) + 1);
                }
            }
        }
        for (uint32_t i = alloc_info.n_in_buf; i < total_bufs; ++i) {
            if (bufs[i] && bufs[i]->va && bufs[i]->size > 0) {
                std::memset(bufs[i]->va, 0, bufs[i]->size);
            }
        }

        const auto e0 = std::chrono::steady_clock::now();
        rc = syms_->EnnExecuteModel(model_id, 0);
        const auto e1 = std::chrono::steady_clock::now();
        rep.exec_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();
        rep.last_ret_code = rc;
        rep.last_ret_name = enn_status_to_string(rc);

        if (rc == 0) {
            rep.executed = true;
            if (alloc_info.n_out_buf > 0) {
                const abi::_ennBuffer* ob = bufs[alloc_info.n_in_buf];
                if (ob && ob->va && ob->size > 0) {
                    const uint8_t* op = static_cast<const uint8_t*>(ob->va);
                    rep.first_output_fnv1a = fnv1a_64(op, ob->size);
                    for (uint32_t b = 0; b < ob->size; ++b) {
                        if (op[b] != 0) {
                            rep.first_output_nonzero = true;
                            break;
                        }
                    }
                }
            }
            probe_report_.status = runtime::CapabilityStatus::EXECUTION_VERIFIED;
        } else {
            rep.error_msg = "EnnExecuteModel failed: " + rep.last_ret_name +
                            " (" + std::to_string(rc) + ")";
        }

        syms_->EnnReleaseBuffers(bufs, static_cast<int>(total_bufs));
    }

    syms_->EnnCloseModel(model_id);
    return rep;
#endif
}

} // namespace npu
} // namespace exynova
