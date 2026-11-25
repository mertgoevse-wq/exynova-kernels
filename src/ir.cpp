#include "exynova/ir.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace exynova {
namespace compiler {

int64_t TensorDesc::element_count() const {
    if (shape.empty()) {
        return 0;
    }
    int64_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= dim;
    }
    return count;
}

size_t TensorDesc::byte_size() const {
    const int64_t elems = element_count();
    if (elems <= 0) {
        return 0;
    }
    return static_cast<size_t>(elems) * dtype_size_bytes(dtype);
}

static const TensorDesc* find_tensor_in_vec(const std::vector<TensorDesc>& tensors,
                                            uint32_t id) {
    for (const auto& t : tensors) {
        if (t.id == id) {
            return &t;
        }
    }
    return nullptr;
}

uint64_t Node::estimated_flops() const {
    return cached_flops;
}

uint64_t Node::estimated_flops(const std::vector<TensorDesc>& tensors) const {
    const TensorDesc* in0 = inputs.empty() ? nullptr : find_tensor_in_vec(tensors, inputs[0]);
    const TensorDesc* in1 = inputs.size() < 2 ? nullptr : find_tensor_in_vec(tensors, inputs[1]);
    const TensorDesc* out0 = outputs.empty() ? nullptr : find_tensor_in_vec(tensors, outputs[0]);

    const uint64_t out_elems =
        out0 ? static_cast<uint64_t>(std::max<int64_t>(0, out0->element_count())) : 0ULL;

    switch (op) {
    case OpType::Add:
    case OpType::Mul:
    case OpType::Relu:
    case OpType::Quantize:
    case OpType::Dequantize:
        return out_elems;
    case OpType::SiLU:
    case OpType::RMSNorm:
        return out_elems * 4ULL;
    case OpType::Softmax:
        return out_elems * 5ULL;
    case OpType::MatMul: {
        if (!in0 || !in1 || in0->shape.size() < 2 || in1->shape.size() < 2) {
            return out_elems * 2ULL;
        }
        const size_t r0 = in0->shape.size();
        const int64_t m = in0->shape[r0 - 2];
        const int64_t k = in0->shape[r0 - 1];
        const int64_t n = in1->shape[in1->shape.size() - 1];
        if (m <= 0 || k <= 0 || n <= 0) {
            return 0;
        }
        int64_t batch = 1;
        if (out0 && out0->shape.size() > 2) {
            for (size_t i = 0; i + 2 < out0->shape.size(); ++i) {
                batch *= std::max<int64_t>(1, out0->shape[i]);
            }
        }
        uint64_t flops = 2ULL * static_cast<uint64_t>(batch) *
                         static_cast<uint64_t>(m) *
                         static_cast<uint64_t>(n) *
                         static_cast<uint64_t>(k);
        if (!attrs.fused_activation.empty()) {
            flops += out_elems;
        }
        return flops;
    }
    case OpType::Conv2D: {
        if (!in0 || !in1 || !out0 || in0->shape.size() != 4 ||
            in1->shape.size() != 4 || out0->shape.size() != 4) {
            return out_elems * 2ULL;
        }
        const bool is_nchw = (in0->layout == Layout::NCHW || out0->layout == Layout::NCHW);
        const int64_t c_in = is_nchw ? in1->shape[1] : in1->shape[3];
        const int64_t kh = is_nchw ? in1->shape[2] : in1->shape[1];
        const int64_t kw = is_nchw ? in1->shape[3] : in1->shape[2];
        if (c_in <= 0 || kh <= 0 || kw <= 0) {
            return 0;
        }
        uint64_t flops = 2ULL * out_elems *
                         static_cast<uint64_t>(c_in) *
                         static_cast<uint64_t>(kh) *
                         static_cast<uint64_t>(kw);
        if (!attrs.fused_activation.empty()) {
            flops += out_elems;
        }
        return flops;
    }
    case OpType::EnnPrecompiled:
        return out_elems > 0 ? out_elems * 2ULL : 1ULL;
    }
    return 0;
}

uint32_t Graph::add_tensor(const std::string& tensor_name,
                           const std::vector<int64_t>& shape,
                           DType dtype,
                           Layout layout,
                           bool is_constant,
                           std::vector<uint8_t> constant_data) {
    TensorDesc desc;
    desc.id = next_tensor_id_++;
    desc.name = tensor_name;
    desc.shape = shape;
    desc.dtype = dtype;
    desc.layout = layout;
    desc.is_constant = is_constant;
    desc.constant_data = std::move(constant_data);
    tensors.push_back(std::move(desc));
    return tensors.back().id;
}

uint32_t Graph::add_constant_f32(const std::string& tensor_name,
                                 const std::vector<int64_t>& shape,
                                 const std::vector<float>& values,
                                 Layout layout) {
    std::vector<uint8_t> raw(values.size() * sizeof(float));
    if (!values.empty()) {
        std::memcpy(raw.data(), values.data(), raw.size());
    }
    return add_tensor(tensor_name, shape, DType::FP32, layout, true, std::move(raw));
}

uint32_t Graph::add_node(const std::string& node_name,
                         OpType op,
                         std::vector<uint32_t> node_inputs,
                         std::vector<uint32_t> node_outputs,
                         NodeAttrs attrs,
                         runtime::BackendKind preferred_backend) {
    Node node;
    node.id = next_node_id_++;
    node.name = node_name;
    node.op = op;
    node.inputs = std::move(node_inputs);
    node.outputs = std::move(node_outputs);
    node.attrs = std::move(attrs);
    node.preferred_backend = preferred_backend;
    node.cached_flops = node.estimated_flops(tensors);
    nodes.push_back(std::move(node));
    return nodes.back().id;
}

const TensorDesc* Graph::get_tensor(uint32_t id) const {
    return find_tensor_in_vec(tensors, id);
}

TensorDesc* Graph::get_tensor_mut(uint32_t id) {
    for (auto& t : tensors) {
        if (t.id == id) {
            return &t;
        }
    }
    return nullptr;
}

const TensorDesc* Graph::find_tensor_by_name(const std::string& tensor_name) const {
    for (const auto& t : tensors) {
        if (t.name == tensor_name) {
            return &t;
        }
    }
    return nullptr;
}

const Node* Graph::get_node(uint32_t id) const {
    for (const auto& n : nodes) {
        if (n.id == id) {
            return &n;
        }
    }
    return nullptr;
}

Node* Graph::get_node_mut(uint32_t id) {
    for (auto& n : nodes) {
        if (n.id == id) {
            return &n;
        }
    }
    return nullptr;
}

void Graph::sync_id_counters() {
    uint32_t max_tid = 0;
    for (const auto& t : tensors) {
        if (t.id > max_tid) {
            max_tid = t.id;
        }
    }
    uint32_t max_nid = 0;
    for (const auto& n : nodes) {
        if (n.id > max_nid) {
            max_nid = n.id;
        }
    }
    next_tensor_id_ = max_tid + 1;
    next_node_id_ = max_nid + 1;
}

std::vector<uint32_t> Graph::topological_order(bool* has_cycle) const {
    if (has_cycle) {
        *has_cycle = false;
    }
    if (nodes.empty()) {
        return {};
    }

    std::unordered_map<uint32_t, size_t> id_to_idx;
    std::unordered_map<uint32_t, size_t> tensor_producer;
    for (size_t i = 0; i < nodes.size(); ++i) {
        id_to_idx[nodes[i].id] = i;
        for (uint32_t out_tid : nodes[i].outputs) {
            tensor_producer[out_tid] = i;
        }
    }

    std::vector<std::vector<size_t>> adj(nodes.size());
    std::vector<size_t> in_degree(nodes.size(), 0);

    for (size_t i = 0; i < nodes.size(); ++i) {
        std::unordered_set<size_t> seen_preds;
        for (uint32_t in_tid : nodes[i].inputs) {
            auto it = tensor_producer.find(in_tid);
            if (it != tensor_producer.end()) {
                const size_t pred_idx = it->second;
                if (seen_preds.insert(pred_idx).second) {
                    adj[pred_idx].push_back(i);
                    ++in_degree[i];
                }
            }
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (in_degree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<uint32_t> order;
    order.reserve(nodes.size());
    while (!q.empty()) {
        const size_t cur = q.front();
        q.pop();
        order.push_back(nodes[cur].id);
        for (size_t nxt : adj[cur]) {
            if (--in_degree[nxt] == 0) {
                q.push(nxt);
            }
        }
    }

    if (order.size() != nodes.size()) {
        if (has_cycle) {
            *has_cycle = true;
        }
        return {};
    }
    return order;
}

bool Graph::validate(std::string* err) const {
    auto fail = [&](const std::string& msg) {
        if (err) {
            *err = msg;
        }
        return false;
    };

    if (ir_version != 1) {
        return fail("Unsupported IR version: " + std::to_string(ir_version));
    }

    std::unordered_set<uint32_t> tensor_ids;
    for (const auto& t : tensors) {
        if (t.id == 0) {
            return fail("Invalid tensor ID 0");
        }
        if (!tensor_ids.insert(t.id).second) {
            return fail("Duplicate tensor ID: " + std::to_string(t.id));
        }
        if (t.is_constant && t.byte_size() > 0 && !t.constant_data.empty() &&
            t.constant_data.size() != t.byte_size()) {
            return fail("Constant tensor '" + t.name + "' data size mismatch");
        }
    }

    for (uint32_t in_id : inputs) {
        if (tensor_ids.find(in_id) == tensor_ids.end()) {
            return fail("Graph input references unknown tensor ID: " + std::to_string(in_id));
        }
    }
    for (uint32_t out_id : outputs) {
        if (tensor_ids.find(out_id) == tensor_ids.end()) {
            return fail("Graph output references unknown tensor ID: " + std::to_string(out_id));
        }
    }

    std::unordered_set<uint32_t> node_ids;
    std::unordered_set<uint32_t> produced_tensors;
    for (const auto& n : nodes) {
        if (n.id == 0) {
            return fail("Invalid node ID 0");
        }
        if (!node_ids.insert(n.id).second) {
            return fail("Duplicate node ID: " + std::to_string(n.id));
        }
        for (uint32_t tid : n.inputs) {
            if (tensor_ids.find(tid) == tensor_ids.end()) {
                return fail("Node '" + n.name + "' references unknown input tensor ID: " +
                            std::to_string(tid));
            }
        }
        for (uint32_t tid : n.outputs) {
            if (tensor_ids.find(tid) == tensor_ids.end()) {
                return fail("Node '" + n.name + "' references unknown output tensor ID: " +
                            std::to_string(tid));
            }
            if (!produced_tensors.insert(tid).second) {
                return fail("Tensor ID " + std::to_string(tid) + " has multiple producers");
            }
        }
    }

    bool has_cycle = false;
    topological_order(&has_cycle);
    if (has_cycle) {
        return fail("Graph contains a cycle");
    }

    if (err) {
        err->clear();
    }
    return true;
}

static bool broadcast_shapes(const std::vector<int64_t>& a,
                             const std::vector<int64_t>& b,
                             std::vector<int64_t>& out,
                             std::string* err) {
    const size_t nd = std::max(a.size(), b.size());
    out.resize(nd);
    for (size_t i = 0; i < nd; ++i) {
        const size_t ia = i + a.size();
        const size_t ib = i + b.size();
        const int64_t da = (ia < nd) ? 1 : a[ia - nd];
        const int64_t db = (ib < nd) ? 1 : b[ib - nd];
        if (da != db && da != 1 && db != 1) {
            if (err) {
                *err = "Incompatible broadcast dimensions";
            }
            return false;
        }
        out[i] = std::max(da, db);
    }
    return true;
}

bool Graph::infer_shapes(std::string* err) {
    if (!validate(err)) {
        return false;
    }

    auto fail = [&](const std::string& msg) {
        if (err) {
            *err = msg;
        }
        return false;
    };

    const std::vector<uint32_t> order = topological_order();
    for (uint32_t nid : order) {
        Node* node = get_node_mut(nid);
        if (!node) {
            continue;
        }

        switch (node->op) {
        case OpType::Add:
        case OpType::Mul: {
            if (node->inputs.size() < 2 || node->outputs.empty()) {
                return fail("Elementwise binary op '" + node->name + "' requires 2 inputs and 1 output");
            }
            const TensorDesc* a = get_tensor(node->inputs[0]);
            const TensorDesc* b = get_tensor(node->inputs[1]);
            TensorDesc* out = get_tensor_mut(node->outputs[0]);
            if (!a || !b || !out) {
                return fail("Missing tensor in node '" + node->name + "'");
            }
            std::vector<int64_t> out_shape;
            if (!broadcast_shapes(a->shape, b->shape, out_shape, err)) {
                return false;
            }
            out->shape = std::move(out_shape);
            out->dtype = a->dtype;
            if (out->layout == Layout::ANY) {
                out->layout = a->layout;
            }
            break;
        }
        case OpType::MatMul: {
            if (node->inputs.size() < 2 || node->outputs.empty()) {
                return fail("MatMul node '" + node->name + "' requires 2 inputs and 1 output");
            }
            const TensorDesc* a = get_tensor(node->inputs[0]);
            const TensorDesc* b = get_tensor(node->inputs[1]);
            TensorDesc* out = get_tensor_mut(node->outputs[0]);
            if (!a || !b || !out) {
                return fail("Missing tensor in MatMul '" + node->name + "'");
            }
            if (a->shape.size() < 2 || b->shape.size() < 2) {
                return fail("MatMul '" + node->name + "' requires rank >= 2 inputs");
            }
            const size_t ra = a->shape.size();
            const size_t rb = b->shape.size();
            const int64_t m = a->shape[ra - 2];
            const int64_t k1 = a->shape[ra - 1];
            const int64_t k2 = b->shape[rb - 2];
            const int64_t n = b->shape[rb - 1];
            if (k1 != k2) {
                return fail("MatMul '" + node->name + "' inner dimension mismatch: " +
                            std::to_string(k1) + " vs " + std::to_string(k2));
            }
            std::vector<int64_t> ba(a->shape.begin(), a->shape.end() - 2);
            std::vector<int64_t> bb(b->shape.begin(), b->shape.end() - 2);
            std::vector<int64_t> out_shape;
            if (!broadcast_shapes(ba, bb, out_shape, err)) {
                return false;
            }
            out_shape.push_back(m);
            out_shape.push_back(n);
            out->shape = std::move(out_shape);
            out->dtype = a->dtype;
            if (out->layout == Layout::ANY && out->shape.size() == 2) {
                out->layout = Layout::NC;
            }
            break;
        }
        case OpType::Conv2D: {
            if (node->inputs.size() < 2 || node->outputs.empty()) {
                return fail("Conv2D node '" + node->name + "' requires >= 2 inputs and 1 output");
            }
            const TensorDesc* x = get_tensor(node->inputs[0]);
            const TensorDesc* w = get_tensor(node->inputs[1]);
            TensorDesc* out = get_tensor_mut(node->outputs[0]);
            if (!x || !w || !out) {
                return fail("Missing tensor in Conv2D '" + node->name + "'");
            }
            if (x->shape.size() != 4 || w->shape.size() != 4) {
                return fail("Conv2D '" + node->name + "' requires 4D input and weight tensors");
            }
            const int32_t sh = std::max(1, node->attrs.strides[0]);
            const int32_t sw = std::max(1, node->attrs.strides[1]);
            const int32_t pt = node->attrs.padding[0];
            const int32_t pb = node->attrs.padding[1];
            const int32_t pl = node->attrs.padding[2];
            const int32_t pr = node->attrs.padding[3];
            const bool is_nchw = (x->layout == Layout::NCHW);
            if (is_nchw) {
                const int64_t n = x->shape[0];
                const int64_t c_in = x->shape[1];
                const int64_t h = x->shape[2];
                const int64_t width = x->shape[3];
                const int64_t c_out = w->shape[0];
                const int64_t c_in_w = w->shape[1];
                const int64_t kh = w->shape[2];
                const int64_t kw = w->shape[3];
                if (c_in != c_in_w) {
                    return fail("Conv2D NCHW channel mismatch in '" + node->name + "'");
                }
                const int64_t h_out = (h + pt + pb - kh) / sh + 1;
                const int64_t w_out = (width + pl + pr - kw) / sw + 1;
                if (h_out <= 0 || w_out <= 0) {
                    return fail("Conv2D invalid output spatial dims in '" + node->name + "'");
                }
                out->shape = {n, c_out, h_out, w_out};
                out->layout = Layout::NCHW;
            } else {
                const int64_t n = x->shape[0];
                const int64_t h = x->shape[1];
                const int64_t width = x->shape[2];
                const int64_t c_in = x->shape[3];
                const int64_t c_out = w->shape[0];
                const int64_t kh = w->shape[1];
                const int64_t kw = w->shape[2];
                const int64_t c_in_w = w->shape[3];
                if (c_in != c_in_w) {
                    return fail("Conv2D NHWC channel mismatch in '" + node->name + "'");
                }
                const int64_t h_out = (h + pt + pb - kh) / sh + 1;
                const int64_t w_out = (width + pl + pr - kw) / sw + 1;
                if (h_out <= 0 || w_out <= 0) {
                    return fail("Conv2D invalid output spatial dims in '" + node->name + "'");
                }
                out->shape = {n, h_out, w_out, c_out};
                out->layout = Layout::NHWC;
            }
            out->dtype = x->dtype;
            break;
        }
        case OpType::Relu:
        case OpType::SiLU:
        case OpType::Softmax:
        case OpType::RMSNorm: {
            if (node->inputs.empty() || node->outputs.empty()) {
                return fail("Unary/norm node '" + node->name + "' missing input or output");
            }
            const TensorDesc* in0 = get_tensor(node->inputs[0]);
            TensorDesc* out = get_tensor_mut(node->outputs[0]);
            if (!in0 || !out) {
                return fail("Missing tensor in node '" + node->name + "'");
            }
            out->shape = in0->shape;
            out->dtype = in0->dtype;
            out->layout = in0->layout;
            break;
        }
        case OpType::Quantize:
        case OpType::Dequantize: {
            if (node->inputs.empty() || node->outputs.empty()) {
                return fail("Quant node '" + node->name + "' missing input or output");
            }
            const TensorDesc* in0 = get_tensor(node->inputs[0]);
            TensorDesc* out = get_tensor_mut(node->outputs[0]);
            if (!in0 || !out) {
                return fail("Missing tensor in node '" + node->name + "'");
            }
            out->shape = in0->shape;
            out->layout = in0->layout;
            break;
        }
        case OpType::EnnPrecompiled:
            break;
        }

        node->cached_flops = node->estimated_flops(tensors);
    }

    if (err) {
        err->clear();
    }
    return true;
}

size_t Graph::run_dead_code_elimination() {
    std::unordered_map<uint32_t, size_t> producer_node_idx;
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (uint32_t out_tid : nodes[i].outputs) {
            producer_node_idx[out_tid] = i;
        }
    }

    std::unordered_set<uint32_t> live_tensors;
    std::unordered_set<uint32_t> live_nodes;
    std::queue<uint32_t> work;

    for (uint32_t out_tid : outputs) {
        if (live_tensors.insert(out_tid).second) {
            work.push(out_tid);
        }
    }

    while (!work.empty()) {
        const uint32_t tid = work.front();
        work.pop();
        auto it = producer_node_idx.find(tid);
        if (it == producer_node_idx.end()) {
            continue;
        }
        const Node& prod = nodes[it->second];
        if (live_nodes.insert(prod.id).second) {
            for (uint32_t out_tid : prod.outputs) {
                live_tensors.insert(out_tid);
            }
            for (uint32_t in_tid : prod.inputs) {
                if (live_tensors.insert(in_tid).second) {
                    work.push(in_tid);
                }
            }
        }
    }

    for (uint32_t in_tid : inputs) {
        live_tensors.insert(in_tid);
    }

    const size_t before_nodes = nodes.size();
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
                       [&](const Node& n) { return live_nodes.find(n.id) == live_nodes.end(); }),
        nodes.end());

    tensors.erase(
        std::remove_if(tensors.begin(), tensors.end(),
                       [&](const TensorDesc& t) {
                           return live_tensors.find(t.id) == live_tensors.end();
                       }),
        tensors.end());

    return before_nodes - nodes.size();
}

size_t Graph::run_operator_fusion() {
    size_t fused_count = 0;
    bool changed = true;

    while (changed) {
        changed = false;
        std::unordered_map<uint32_t, size_t> consumer_count;
        std::unordered_map<uint32_t, size_t> producer_idx;
        std::unordered_set<uint32_t> graph_outputs(outputs.begin(), outputs.end());

        for (size_t i = 0; i < nodes.size(); ++i) {
            for (uint32_t tid : nodes[i].inputs) {
                ++consumer_count[tid];
            }
            for (uint32_t tid : nodes[i].outputs) {
                producer_idx[tid] = i;
            }
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& relu = nodes[i];
            if (relu.op != OpType::Relu || relu.inputs.size() != 1 || relu.outputs.size() != 1) {
                continue;
            }
            const uint32_t mid_tid = relu.inputs[0];
            if (graph_outputs.find(mid_tid) != graph_outputs.end()) {
                continue;
            }
            if (consumer_count[mid_tid] != 1) {
                continue;
            }
            auto pit = producer_idx.find(mid_tid);
            if (pit == producer_idx.end()) {
                continue;
            }
            Node& prod = nodes[pit->second];
            if ((prod.op != OpType::MatMul && prod.op != OpType::Conv2D) ||
                !prod.attrs.fused_activation.empty() || prod.outputs.size() != 1) {
                continue;
            }

            prod.attrs.fused_activation = "Relu";
            prod.outputs[0] = relu.outputs[0];
            prod.cached_flops = prod.estimated_flops(tensors);

            nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(i));
            tensors.erase(
                std::remove_if(tensors.begin(), tensors.end(),
                               [&](const TensorDesc& t) { return t.id == mid_tid; }),
                tensors.end());

            ++fused_count;
            changed = true;
            break;
        }
    }

    return fused_count;
}

MemoryPlan Graph::compute_memory_plan() const {
    MemoryPlan plan;
    const std::vector<uint32_t> order = topological_order();
    const size_t num_steps = order.size();

    std::unordered_map<uint32_t, size_t> node_step;
    for (size_t s = 0; s < order.size(); ++s) {
        node_step[order[s]] = s;
    }

    std::unordered_set<uint32_t> graph_inputs(inputs.begin(), inputs.end());
    std::unordered_set<uint32_t> graph_outputs(outputs.begin(), outputs.end());

    struct Lifetime {
        uint32_t tid = 0;
        size_t bytes = 0;
        size_t birth = 0;
        size_t death = 0;
    };

    std::unordered_map<uint32_t, Lifetime> lifetimes;
    for (const auto& t : tensors) {
        if (t.is_constant) {
            continue;
        }
        const size_t bytes = t.byte_size();
        if (bytes == 0) {
            continue;
        }
        Lifetime lt;
        lt.tid = t.id;
        lt.bytes = bytes;
        lt.birth = 0;
        lt.death = (graph_outputs.find(t.id) != graph_outputs.end()) ? num_steps : 0;
        lifetimes[t.id] = lt;
    }

    for (const auto& n : nodes) {
        auto sit = node_step.find(n.id);
        const size_t step = (sit != node_step.end()) ? sit->second : 0;
        for (uint32_t out_tid : n.outputs) {
            auto lit = lifetimes.find(out_tid);
            if (lit != lifetimes.end() && graph_inputs.find(out_tid) == graph_inputs.end()) {
                lit->second.birth = step;
                if (lit->second.death < step) {
                    lit->second.death = step;
                }
            }
        }
        for (uint32_t in_tid : n.inputs) {
            auto lit = lifetimes.find(in_tid);
            if (lit != lifetimes.end()) {
                if (step > lit->second.death) {
                    lit->second.death = step;
                }
            }
        }
    }

    std::vector<Lifetime> sorted;
    sorted.reserve(lifetimes.size());
    for (const auto& kv : lifetimes) {
        sorted.push_back(kv.second);
        plan.total_unshared_bytes += kv.second.bytes;
    }

    std::sort(sorted.begin(), sorted.end(), [](const Lifetime& a, const Lifetime& b) {
        if (a.birth != b.birth) {
            return a.birth < b.birth;
        }
        return a.tid < b.tid;
    });

    struct BufferSlot {
        size_t capacity = 0;
        size_t free_after = 0;
    };
    std::vector<BufferSlot> pool;

    for (const auto& lt : sorted) {
        size_t best_fit_idx = SIZE_MAX;
        size_t largest_free_idx = SIZE_MAX;

        for (size_t i = 0; i < pool.size(); ++i) {
            if (pool[i].free_after < lt.birth) {
                if (pool[i].capacity >= lt.bytes) {
                    if (best_fit_idx == SIZE_MAX || pool[i].capacity < pool[best_fit_idx].capacity) {
                        best_fit_idx = i;
                    }
                }
                if (largest_free_idx == SIZE_MAX ||
                    pool[i].capacity > pool[largest_free_idx].capacity) {
                    largest_free_idx = i;
                }
            }
        }

        if (best_fit_idx != SIZE_MAX) {
            pool[best_fit_idx].free_after = lt.death;
            ++plan.buffer_reuse_count;
            plan.tensor_to_buffer.emplace_back(lt.tid, best_fit_idx);
        } else if (largest_free_idx != SIZE_MAX) {
            pool[largest_free_idx].capacity = std::max(pool[largest_free_idx].capacity, lt.bytes);
            pool[largest_free_idx].free_after = lt.death;
            ++plan.buffer_reuse_count;
            plan.tensor_to_buffer.emplace_back(lt.tid, largest_free_idx);
        } else {
            const size_t idx = pool.size();
            pool.push_back(BufferSlot{lt.bytes, lt.death});
            plan.tensor_to_buffer.emplace_back(lt.tid, idx);
        }
    }

    plan.allocated_buffers = pool.size();
    for (const auto& slot : pool) {
        plan.peak_bytes += slot.capacity;
    }
    return plan;
}

std::string Graph::dump_summary() const {
    std::ostringstream oss;
    oss << "Graph '" << name << "' (IR v" << ir_version << "): "
        << tensors.size() << " tensors, " << nodes.size() << " nodes\n";
    for (uint32_t nid : topological_order()) {
        const Node* n = get_node(nid);
        if (!n) {
            continue;
        }
        oss << "  [" << n->id << "] " << n->name << " : " << to_string(n->op);
        if (!n->attrs.fused_activation.empty()) {
            oss << "+" << n->attrs.fused_activation;
        }
        oss << " (flops=" << n->estimated_flops(tensors) << ")\n";
    }
    return oss.str();
}

} // namespace compiler
} // namespace exynova
