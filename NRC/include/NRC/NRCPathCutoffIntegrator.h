#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "FunctionLayer/Integrator/PathIntegrator-new.h"
#include "NRC/NRCBatchPredictor.h"

namespace NRC {

class NRCPathCutoffIntegrator : public PathIntegratorNew {
public:
    NRCPathCutoffIntegrator(std::shared_ptr<Camera> camera,
                            std::unique_ptr<Film> film,
                            std::shared_ptr<Sampler> sampler,
                            int spp,
                            int maxPtBounce,
                            NRCBatchPredictor predictor,
                            std::filesystem::path tempDir,
                            float energyGain = 1.0f);

    void render(std::shared_ptr<Scene> scene) override;
    Spectrum Li(const Ray &ray, std::shared_ptr<Scene> scene) override;

private:
    struct PendingQuery {
        Point2i pixel = {0, 0};
        Spectrum throughput{0.0};
        std::array<float, 6> feature{};
    };

    Spectrum traceSampleWithCutoff(const Ray &initialRay,
                                   std::shared_ptr<Scene> scene,
                                   const Point2i &pixel);

    void flushNrcQueries();
    void finalizeFilmFromAccumulation();

    int maxPtBounce_ = 4;
    float energyGain_ = 1.0f;
    NRCBatchPredictor predictor_;
    std::filesystem::path tempDir_;

    std::vector<Spectrum> pixelSums_;
    std::vector<uint32_t> pixelSamples_;
    std::vector<PendingQuery> pendingQueries_;
};

}  // namespace NRC
