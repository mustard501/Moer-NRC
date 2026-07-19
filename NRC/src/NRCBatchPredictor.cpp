#include "NRC/NRCBatchPredictor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace NRC {

namespace {

std::filesystem::path toAbsolute(const std::filesystem::path &path) {
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::absolute(path);
}

std::array<float, 3> readVec3FromJson(const Json &json, const std::string &key) {
    if (!json.contains(key) || !json[key].is_array() || json[key].size() < 3) {
        throw std::runtime_error("Missing or invalid vec3 field in train meta: " + key);
    }
    return {
        json[key][0].get<float>(),
        json[key][1].get<float>(),
        json[key][2].get<float>(),
    };
}

Vec3d normalizeVec3(const Vec3d &v) {
    const double len2 = dot(v, v);
    if (len2 <= 1e-20) {
        return Vec3d(0.0, 0.0, 1.0);
    }
    return v / std::sqrt(len2);
}

}  // namespace

NRCBatchPredictor::NRCBatchPredictor(NRCBatchPredictorConfig config)
    : config_(std::move(config)) {
    if (config_.checkpointPath.empty()) {
        throw std::runtime_error("NRC checkpoint path is empty");
    }
    if (config_.trainMetaPath.empty()) {
        throw std::runtime_error("NRC train meta path is empty");
    }
    config_.checkpointPath = toAbsolute(config_.checkpointPath);
    config_.trainMetaPath = toAbsolute(config_.trainMetaPath);
    config_.inferScriptPath = toAbsolute(config_.inferScriptPath);
    loadMetaAabb();
}

void NRCBatchPredictor::loadMetaAabb() {
    std::ifstream in(config_.trainMetaPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open train meta: " + config_.trainMetaPath.string());
    }
    Json meta;
    in >> meta;
    aabbMin_ = readVec3FromJson(meta, "aabb_min");
    aabbMax_ = readVec3FromJson(meta, "aabb_max");
}

bool NRCBatchPredictor::insideAabb(const Point3d &p) const {
    return p.x >= aabbMin_[0] && p.x <= aabbMax_[0] && p.y >= aabbMin_[1]
        && p.y <= aabbMax_[1] && p.z >= aabbMin_[2] && p.z <= aabbMax_[2];
}

std::array<float, 6> NRCBatchPredictor::buildFeature(const Point3d &position,
                                                     const Vec3d &wo) const {
    std::array<float, 6> feature{};
    for (int i = 0; i < 3; ++i) {
        const float p = static_cast<float>(i == 0 ? position.x : (i == 1 ? position.y : position.z));
        const float minv = aabbMin_[i];
        const float maxv = aabbMax_[i];
        const float extent = std::max(maxv - minv, 1e-8f);
        feature[i] = (p - minv) / extent;
    }

    // Keep direction in [0,1] to match current training pipeline/config.
    const Vec3d woN = normalizeVec3(wo);
    feature[3] = static_cast<float>(0.5 * (woN.x + 1.0));
    feature[4] = static_cast<float>(0.5 * (woN.y + 1.0));
    feature[5] = static_cast<float>(0.5 * (woN.z + 1.0));

    // const Vec3d nN = normalizeNormal(normal);
    // feature[6] = static_cast<float>(0.5 * (nN.x + 1.0));
    // feature[7] = static_cast<float>(0.5 * (nN.y + 1.0));
    // feature[8] = static_cast<float>(0.5 * (nN.z + 1.0));

    return feature;
}

std::vector<std::array<float, 3>> NRCBatchPredictor::predict(
    const std::vector<std::array<float, 6>> &features,
    const std::filesystem::path &workingDir,
    const std::string &prefix) const {
    std::vector<std::array<float, 3>> empty;
    if (features.empty()) {
        return empty;
    }

    std::filesystem::create_directories(workingDir);
    const std::filesystem::path featureBin = workingDir / (prefix + "_features.bin");
    const std::filesystem::path outputBin = workingDir / (prefix + "_pred.bin");
    writeFeatureBin(featureBin, features);

    const std::string cmd = buildPythonCommand(featureBin, outputBin);
    const int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("NRC Python inference failed, command: " + cmd);
    }
    return readPredictionBin(outputBin);
}

void NRCBatchPredictor::writeFeatureBin(const std::filesystem::path &path,
                                        const std::vector<std::array<float, 6>> &features) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open feature bin for write: " + path.string());
    }

    const uint64_t n = static_cast<uint64_t>(features.size());
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &f : features) {
        out.write(reinterpret_cast<const char *>(f.data()), sizeof(float) * f.size());
    }
    out.flush();
    if (!out.good()) {
        throw std::runtime_error("Failed to write feature bin: " + path.string());
    }
}

std::vector<std::array<float, 3>> NRCBatchPredictor::readPredictionBin(const std::filesystem::path &path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open prediction bin: " + path.string());
    }

    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in.good()) {
        throw std::runtime_error("Failed to read prediction count from: " + path.string());
    }

    std::vector<std::array<float, 3>> out(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        in.read(reinterpret_cast<char *>(out[static_cast<size_t>(i)].data()), sizeof(float) * 3);
        if (!in.good()) {
            throw std::runtime_error("Prediction bin size mismatch: " + path.string());
        }
    }
    return out;
}

std::string NRCBatchPredictor::buildPythonCommand(const std::filesystem::path &featureBin,
                                                  const std::filesystem::path &outputBin) const {
    std::ostringstream oss;
    oss << config_.pythonCommand
        << " \"" << config_.inferScriptPath.string() << "\""
        << " --checkpoint \"" << config_.checkpointPath.string() << "\""
        << " --features-bin \"" << featureBin.string() << "\""
        << " --output-bin \"" << outputBin.string() << "\""
        << " --batch-size " << config_.inferBatchSize;
    return oss.str();
}

}  // namespace NRC
