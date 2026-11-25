#pragma once

#include "exynova/artifact.h"
#include "exynova/backend_registry.h"
#include "exynova/ir.h"
#include "exynova/partitioner.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace exynova {
namespace sdk {

class CompiledModel;
class Session;

class BackendExecutor {
public:
    virtual ~BackendExecutor() = default;
    virtual runtime::BackendKind kind() const = 0;
    virtual runtime::BackendInfo info() const = 0;
    virtual bool supports_op(const compiler::Node& node,
                             const compiler::Graph& graph) const = 0;
    virtual bool execute_node(const compiler::Node& node,
                              const compiler::Graph& graph,
                              const std::vector<const void*>& inputs,
                              const std::vector<void*>& outputs,
                              std::string* err) = 0;
};

class Context {
public:
    explicit Context(const compiler::TargetHardwareProfile& profile);

    static std::unique_ptr<Context> create(
        const compiler::TargetHardwareProfile& profile =
            compiler::TargetHardwareProfile::exynos_1580(false, false));

    void register_executor(std::shared_ptr<BackendExecutor> executor);
    std::shared_ptr<BackendExecutor> get_executor(runtime::BackendKind kind) const;

    const compiler::TargetHardwareProfile& hardware_profile() const { return profile_; }
    std::vector<runtime::BackendInfo> list_backends() const;

    bool compile_artifact(const compiler::Artifact& artifact,
                          runtime::ExecutionMode mode,
                          bool strict_mode,
                          std::unique_ptr<CompiledModel>& out_model,
                          std::string* err = nullptr) const;

    bool compile_artifact_bytes(const uint8_t* data,
                                size_t size,
                                runtime::ExecutionMode mode,
                                bool strict_mode,
                                std::unique_ptr<CompiledModel>& out_model,
                                std::string* err = nullptr) const;

    bool load_artifact_file(const std::string& path,
                            std::unique_ptr<CompiledModel>& out_model,
                            std::string* err = nullptr) const;

    bool load_artifact_bytes(const uint8_t* data,
                             size_t size,
                             std::unique_ptr<CompiledModel>& out_model,
                             std::string* err = nullptr) const;

private:
    compiler::TargetHardwareProfile profile_;
    std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors_;
};

class CompiledModel {
public:
    CompiledModel(compiler::Artifact artifact,
                  const compiler::TargetHardwareProfile& profile,
                  std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors);

    const compiler::Artifact& artifact() const { return artifact_; }
    const compiler::Graph& graph() const { return artifact_.graph; }
    const compiler::ExecutionPlan& execution_plan() const { return artifact_.plan; }
    const compiler::TargetHardwareProfile& hardware_profile() const { return profile_; }

    std::unique_ptr<Session> create_session() const;

private:
    compiler::Artifact artifact_;
    compiler::TargetHardwareProfile profile_;
    std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors_;
};

class Session {
public:
    Session(const compiler::Artifact& artifact,
            const compiler::TargetHardwareProfile& profile,
            std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors);

    bool set_input(const std::string& tensor_name,
                   const void* data,
                   size_t byte_len,
                   std::string* err = nullptr);

    bool set_input(const std::string& tensor_name,
                   const std::vector<float>& data,
                   std::string* err = nullptr);

    bool run(std::string* err = nullptr);

    bool get_output(const std::string& tensor_name,
                    void* out_data,
                    size_t byte_len,
                    std::string* err = nullptr) const;

    bool get_output(const std::string& tensor_name,
                    std::vector<float>& out_data,
                    std::string* err = nullptr) const;

    const runtime::ExecutionReport& last_report() const { return last_report_; }

private:
    compiler::Artifact artifact_;
    compiler::TargetHardwareProfile profile_;
    std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> tensor_buffers_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
    runtime::ExecutionReport last_report_;

    void init_buffers();
};

const char* version_string();

} // namespace sdk
} // namespace exynova
