#include "NRC/NRCPathCutoffIntegrator.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "FunctionLayer/TileGenerator/SequenceTileGenerator.h"

namespace NRC {

NRCPathCutoffIntegrator::NRCPathCutoffIntegrator(std::shared_ptr<Camera> camera,
                                                 std::unique_ptr<Film> film,
                                                 std::shared_ptr<Sampler> sampler,
                                                 int spp,
                                                 int maxPtBounce,
                                                 NRCBatchPredictor predictor,
                                                 std::filesystem::path tempDir,
                                                 float energyGain)
    : PathIntegratorNew(std::move(camera),
                        std::move(film),
                        std::make_unique<SequenceTileGenerator>(Point2i(1, 1)),
                        std::move(sampler),
                        spp,
                        1),
      maxPtBounce_(maxPtBounce),
      energyGain_(energyGain),
      predictor_(std::move(predictor)),
      tempDir_(std::move(tempDir)) {
    if (maxPtBounce_ < 0) {
        throw std::runtime_error("maxPtBounce must be >= 0");
    }
}

void NRCPathCutoffIntegrator::render(std::shared_ptr<Scene> scene) {
    const Point2i resolution = film->getResolution();
    const int width = resolution.x;
    const int height = resolution.y;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid film resolution");
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    pixelSums_.assign(pixelCount, Spectrum(0.0));
    pixelSamples_.assign(pixelCount, 0);
    pendingQueries_.clear();
    pendingQueries_.reserve(pixelCount * static_cast<size_t>(std::max(1, spp / 2)));

    sampler->startPixel(Point2i(0, 0));
    auto localSampler = sampler->clone(0);

    uint64_t finished = 0;
    const uint64_t total = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Point2i pixel(x, y);
            const size_t pixelId = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            localSampler->startPixel(pixel);

            for (int s = 0; s < spp; ++s) {
                const Ray ray = camera->generateRay(resolution, pixel, localSampler->getCameraSample());
                const Spectrum base = traceSampleWithCutoff(ray, scene, pixel);
                pixelSums_[pixelId] += base;
                pixelSamples_[pixelId] += 1;
                localSampler->nextSample();
            }
            ++finished;
        }
        printProgress(static_cast<float>(finished) / static_cast<float>(total));
    }

    flushNrcQueries();
    finalizeFilmFromAccumulation();
    printProgress(1.f);
}

Spectrum NRCPathCutoffIntegrator::Li(const Ray &ray, std::shared_ptr<Scene> scene) {
    // This integrator renders through traceSampleWithCutoff() in render(), but we keep
    // Li() implemented for compatibility with base class expectations.
    return traceSampleWithCutoff(ray, std::move(scene), Point2i(0, 0));
}

Spectrum NRCPathCutoffIntegrator::traceSampleWithCutoff(const Ray &initialRay,
                                                        std::shared_ptr<Scene> scene,
                                                        const Point2i &pixel) {
    const double eps = 1e-4;
    Spectrum L{0.0};
    Spectrum throughput{1.0};
    Ray ray = initialRay;
    int nBounces = 0;
    auto itsOpt = scene->intersect(ray);

    while (true) {
        if (nBounces == 0) {
            auto evalLightRecord = evalEmittance(scene, itsOpt, ray);
            L += throughput * evalLightRecord.f;
        }

        if (!itsOpt.has_value()) {
            break;
        }

        auto its = itsOpt.value();
        nBounces++;

        if (its.material->getBxDF(its)->isNull()) {
            nBounces--;
            ray = Ray{its.position + ray.direction * eps, ray.direction};
            itsOpt = scene->intersect(ray);
            continue;
        }

        // maxPtBounce semantics:
        //   - 0: keep only direct lighting at the first valid surface hit.
        //   - k>0: keep PT direct-light terms up to bounce k, and use NRC from bounce k+1.
        if (maxPtBounce_ == 0 && nBounces == 1) {
            for (int i = 0; i < nDirectLightSamples; ++i) {
                auto sampleLightRecord = sampleDirectLighting(scene, its, ray);
                auto evalScatterRecord = evalScatter(its, ray, sampleLightRecord.wi);
                if (!sampleLightRecord.f.isBlack()) {
                    double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                    if (sampleLightRecord.isDelta) {
                        misw = 1.0;
                    }
                    L += throughput * sampleLightRecord.f * evalScatterRecord.f / sampleLightRecord.pdf * misw
                        / nDirectLightSamples;
                }
            }
            break;
        }

        // Query NRC at the first bounce beyond PT budget. Query BEFORE adding this bounce's
        // explicit direct-light term, since the NRC target is full L_o(x, wo) at this state.
        if (maxPtBounce_ > 0 && nBounces > maxPtBounce_ && predictor_.insideAabb(its.position)) {
            PendingQuery q;
            q.pixel = pixel;
            q.throughput = throughput;
            q.feature = predictor_.buildFeature(its.position, normalize(-ray.direction));
            pendingQueries_.push_back(q);
            break;
        }

        for (int i = 0; i < nDirectLightSamples; ++i) {
            auto sampleLightRecord = sampleDirectLighting(scene, its, ray);
            auto evalScatterRecord = evalScatter(its, ray, sampleLightRecord.wi);
            if (!sampleLightRecord.f.isBlack()) {
                double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                if (sampleLightRecord.isDelta) {
                    misw = 1.0;
                }
                L += throughput * sampleLightRecord.f * evalScatterRecord.f / sampleLightRecord.pdf * misw
                    / nDirectLightSamples;
            }
        }

        const double pSurvive = russianRoulette(throughput, nBounces);
        if (sampler->sample1D() >= pSurvive) {
            break;
        }
        throughput /= pSurvive;

        auto sampleScatterRecord = sampleScatter(its, ray);
        if (sampleScatterRecord.f.isBlack() || sampleScatterRecord.pdf == 0) {
            break;
        }
        throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;

        ray = Ray{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
        itsOpt = scene->intersect(ray);

        auto evalLightRecord = evalEmittance(scene, itsOpt, ray);
        if (!evalLightRecord.f.isBlack()) {
            double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
            if (sampleScatterRecord.isDelta) {
                misw = 1.0;
            }
            L += throughput * evalLightRecord.f * misw;
        }
    }

    return L;
}

void NRCPathCutoffIntegrator::flushNrcQueries() {
    if (pendingQueries_.empty()) {
        std::cout << "\n[NRC] no cutoff queries generated; skip NRC prediction." << std::endl;
        return;
    }

    std::vector<std::array<float, 6>> features;
    features.reserve(pendingQueries_.size());
    for (const auto &q : pendingQueries_) {
        features.push_back(q.feature);
    }
    auto preds = predictor_.predict(features, tempDir_, "nrc_view");
    if (preds.size() != pendingQueries_.size()) {
        throw std::runtime_error("NRC prediction size mismatch");
    }

    const int width = film->getResolution().x;
    for (size_t i = 0; i < pendingQueries_.size(); ++i) {
        const auto &q = pendingQueries_[i];
        const auto &p = preds[i];
        const RGB3 predRgb(
            std::max(0.0, static_cast<double>(p[0])),
            std::max(0.0, static_cast<double>(p[1])),
            std::max(0.0, static_cast<double>(p[2])));
        const Spectrum predL(predRgb);
        const Spectrum contribution = q.throughput * predL * static_cast<double>(energyGain_);
        const size_t pixelId = static_cast<size_t>(q.pixel.y) * static_cast<size_t>(width)
            + static_cast<size_t>(q.pixel.x);
        pixelSums_[pixelId] += contribution;
    }

    std::cout << "\n[NRC] queried " << pendingQueries_.size() << " cutoff states." << std::endl;
}

void NRCPathCutoffIntegrator::finalizeFilmFromAccumulation() {
    const Point2i resolution = film->getResolution();
    const int width = resolution.x;
    const int height = resolution.y;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t id = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const uint32_t n = pixelSamples_[id];
            if (n == 0) {
                film->deposit(Point2i(x, y), Spectrum(0.0));
                continue;
            }
            film->deposit(Point2i(x, y), pixelSums_[id] / static_cast<double>(n));
        }
    }
}

}  // namespace NRC
