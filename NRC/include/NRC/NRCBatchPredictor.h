#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "CoreLayer/Adapter/JsonUtil.h"
#include "CoreLayer/ColorSpace/Color.h"
#include "CoreLayer/Geometry/Geometry.h"

namespace NRC {

struct NRCBatchPredictorConfig {
    std::filesystem::path checkpointPath;
    std::filesystem::path trainMetaPath;
    std::filesystem::path inferScriptPath = std::filesystem::path("NRC/python/infer_nrc_batch.py");
    std::string pythonCommand = "conda run -n moer-build python";
    size_t inferBatchSize = 1 << 20;
};

class NRCBatchPredictor {
public:
    explicit NRCBatchPredictor(NRCBatchPredictorConfig config);

    [[nodiscard]]
    const std::array<float, 3> &aabbMin() const { return aabbMin_; }

    [[nodiscard]]
    const std::array<float, 3> &aabbMax() const { return aabbMax_; }

    [[nodiscard]]
    bool insideAabb(const Point3d &p) const;

    [[nodiscard]]
    std::array<float, 6> buildFeature(const Point3d &position,
                                      const Vec3d &wo) const;

    [[nodiscard]]
    std::vector<std::array<float, 3>> predict(const std::vector<std::array<float, 6>> &features,
                                              const std::filesystem::path &workingDir,
                                              const std::string &prefix = "nrc_infer") const;

private:
    void loadMetaAabb();

    void writeFeatureBin(const std::filesystem::path &path,
                         const std::vector<std::array<float, 6>> &features) const;

    std::vector<std::array<float, 3>> readPredictionBin(const std::filesystem::path &path) const;

    [[nodiscard]]
    std::string buildPythonCommand(const std::filesystem::path &featureBin,
                                   const std::filesystem::path &outputBin) const;

    NRCBatchPredictorConfig config_;
    std::array<float, 3> aabbMin_ = {0.f, 0.f, 0.f};
    std::array<float, 3> aabbMax_ = {0.f, 0.f, 0.f};
};

}  // namespace NRC
