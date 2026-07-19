#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

#include "FunctionLayer/Integrator/PathIntegrator-new.h"
#include "NRC/NRCDatasetWriter.h"
#include "NRC/NRCTypes.h"

namespace NRC {

class NRCCollectorIntegrator : public PathIntegratorNew {
public:
    NRCCollectorIntegrator(std::shared_ptr<Camera> _camera,
                           std::unique_ptr<Film> _film,
                           std::unique_ptr<TileGenerator> _tileGenerator,
                           std::shared_ptr<Sampler> _sampler,
                           int _spp,
                           const std::string &datasetPath,
                           int _renderThreadNum = 4,
                           size_t flushThreshold = 1 << 15,
                           SampleFilterConfig filterConfig = {},
                           bool useFixedAabb = false,
                           std::array<float, 3> fixedAabbMin = {0.f, 0.f, 0.f},
                           std::array<float, 3> fixedAabbMax = {0.f, 0.f, 0.f});

    void render(std::shared_ptr<Scene> scene) override;
    Spectrum Li(const Ray &initialRay, std::shared_ptr<Scene> scene) override;

    [[nodiscard]]
    uint64_t collectedSamples() const { return sampleCount_.load(); }

    [[nodiscard]]
    uint64_t candidateSamples() const { return candidateCount_.load(); }

    [[nodiscard]]
    uint64_t outOfAabbDroppedSamples() const { return outOfAabbDroppedCount_.load(); }

    // bounceCounts[b] = number of written samples with bounce == b.
    // Index 0 is unused (bounce starts at 1).
    [[nodiscard]]
    const std::vector<uint64_t> &bounceCounts() const { return bounceCounts_; }

    // candidateBounceCounts[b] = samples considered before keep/drop.
    [[nodiscard]]
    const std::vector<uint64_t> &candidateBounceCounts() const { return candidateBounceCounts_; }

    [[nodiscard]]
    const SampleFilterConfig &filterConfig() const { return filterConfig_; }

private:
    struct BounceInfo {
        Spectrum localRadiance{0.0};
        Spectrum bsdfWeight{0.0};
        Point3d position{0.0, 0.0, 0.0};
        Vec3d wo{0.0, 0.0, 1.0};
        Normal3d normal{0.0, 0.0, 1.0};
        uint16_t bounce = 0;
        bool hasBsdfWeight = false;
    };

    void appendSample(const BounceInfo &info);
    void flushPendingSamples();
    void bumpBounceCount(std::vector<uint64_t> &counts, uint16_t bounce);
    bool insideFixedAabb(const Point3d &p) const;

    std::string datasetPath_;
    size_t flushThreshold_;
    SampleFilterConfig filterConfig_;
    bool useFixedAabb_ = false;
    std::array<float, 3> fixedAabbMin_ = {0.f, 0.f, 0.f};
    std::array<float, 3> fixedAabbMax_ = {0.f, 0.f, 0.f};

    std::mutex sampleMutex_;
    std::vector<RadianceSampleRecordV1> pendingSamples_;

    NRCDatasetWriter writer_;
    std::atomic<uint64_t> sampleCount_{0};
    std::atomic<uint64_t> candidateCount_{0};
    std::atomic<uint64_t> outOfAabbDroppedCount_{0};
    std::vector<uint64_t> bounceCounts_;
    std::vector<uint64_t> candidateBounceCounts_;
};

}  // namespace NRC
