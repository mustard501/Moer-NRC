#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace NRC {

constexpr uint32_t kDatasetVersion = 1;
constexpr std::array<char, 4> kDatasetMagic = {'N', 'R', 'C', '1'};

enum DatasetFlags : uint32_t {
    DatasetFlagNone = 0,
    DatasetFlagHasNormal = 1 << 0,
};

struct DatasetHeaderV1 {
    std::array<char, 4> magic{kDatasetMagic};
    uint32_t version = kDatasetVersion;
    uint32_t headerBytes = sizeof(DatasetHeaderV1);
    uint64_t recordCount = 0;
    uint32_t featureDim = 9;  // position(3) + outgoing_dir(3) + normal(3)
    uint32_t labelDim = 3;    // RGB radiance
    uint32_t flags = DatasetFlagHasNormal;
    uint32_t reserved0 = 0;
    float sceneAabbMin[3] = {0.f, 0.f, 0.f};
    float sceneAabbMax[3] = {0.f, 0.f, 0.f};
    uint32_t reserved[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

struct RadianceSampleRecordV1 {
    float position[3];     // world-space point
    float outgoingDir[3];  // normalized wo (towards previous vertex/camera)
    float normal[3];       // shading or geometry normal
    float targetRGB[3];    // outgoing radiance RGB
    uint16_t bounce = 0;   // path depth at this vertex
    uint16_t _padding = 0;
};

/// Write-time keep/drop filter. bounceKeepProbs[0] applies to bounce==1.
struct SampleFilterConfig {
    int maxBounceRecord = 8;
    double globalKeep = 1.0;
    // Default: keep bounce1 fully, taper deeper bounces, drop beyond maxBounceRecord.
    std::vector<double> bounceKeepProbs = {1.0, 0.5, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1};

    [[nodiscard]]
    double keepProbability(uint16_t bounce) const {
        if (bounce == 0 || bounce > static_cast<uint16_t>(maxBounceRecord)) {
            return 0.0;
        }
        const size_t index = static_cast<size_t>(bounce) - 1;
        if (index >= bounceKeepProbs.size()) {
            return 0.0;
        }
        const double p = bounceKeepProbs[index] * globalKeep;
        if (p <= 0.0) {
            return 0.0;
        }
        if (p >= 1.0) {
            return 1.0;
        }
        return p;
    }
};

}  // namespace NRC
