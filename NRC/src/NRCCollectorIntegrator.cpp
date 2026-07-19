#include "NRC/NRCCollectorIntegrator.h"
#include "CoreLayer/ColorSpace/Color.h"

#include <iostream>
#include <stdexcept>

namespace NRC {

NRCCollectorIntegrator::NRCCollectorIntegrator(std::shared_ptr<Camera> _camera,
                                               std::unique_ptr<Film> _film,
                                               std::unique_ptr<TileGenerator> _tileGenerator,
                                               std::shared_ptr<Sampler> _sampler,
                                               int _spp,
                                               const std::string &datasetPath,
                                               int _renderThreadNum,
                                               size_t flushThreshold,
                                               SampleFilterConfig filterConfig,
                                               bool useFixedAabb,
                                               std::array<float, 3> fixedAabbMin,
                                               std::array<float, 3> fixedAabbMax)
    : PathIntegratorNew(std::move(_camera),
                        std::move(_film),
                        std::move(_tileGenerator),
                        std::move(_sampler),
                        _spp,
                        _renderThreadNum),
      datasetPath_(datasetPath),
      flushThreshold_(flushThreshold),
      filterConfig_(std::move(filterConfig)),
      useFixedAabb_(useFixedAabb),
      fixedAabbMin_(fixedAabbMin),
      fixedAabbMax_(fixedAabbMax) {
    pendingSamples_.reserve(flushThreshold_);
}

void NRCCollectorIntegrator::bumpBounceCount(std::vector<uint64_t> &counts, uint16_t bounce) {
    if (bounce >= counts.size()) {
        counts.resize(static_cast<size_t>(bounce) + 1, 0);
    }
    ++counts[bounce];
}

bool NRCCollectorIntegrator::insideFixedAabb(const Point3d &p) const {
    if (!useFixedAabb_) {
        return true;
    }
    return p.x >= fixedAabbMin_[0] && p.x <= fixedAabbMax_[0]
        && p.y >= fixedAabbMin_[1] && p.y <= fixedAabbMax_[1]
        && p.z >= fixedAabbMin_[2] && p.z <= fixedAabbMax_[2];
}

void NRCCollectorIntegrator::render(std::shared_ptr<Scene> scene) {
    DatasetHeaderV1 header{};
    if (useFixedAabb_) {
        header.sceneAabbMin[0] = fixedAabbMin_[0];
        header.sceneAabbMin[1] = fixedAabbMin_[1];
        header.sceneAabbMin[2] = fixedAabbMin_[2];
        header.sceneAabbMax[0] = fixedAabbMax_[0];
        header.sceneAabbMax[1] = fixedAabbMax_[1];
        header.sceneAabbMax[2] = fixedAabbMax_[2];
    } else {
        auto sceneBounds = scene->getGlobalBoundingBox();
        header.sceneAabbMin[0] = static_cast<float>(sceneBounds.pMin.x);
        header.sceneAabbMin[1] = static_cast<float>(sceneBounds.pMin.y);
        header.sceneAabbMin[2] = static_cast<float>(sceneBounds.pMin.z);
        header.sceneAabbMax[0] = static_cast<float>(sceneBounds.pMax.x);
        header.sceneAabbMax[1] = static_cast<float>(sceneBounds.pMax.y);
        header.sceneAabbMax[2] = static_cast<float>(sceneBounds.pMax.z);
    }

    if (!writer_.open(datasetPath_, header)) {
        throw std::runtime_error("Failed to open NRC dataset file: " + datasetPath_);
    }

    MonteCarloIntegrator::render(scene);

    flushPendingSamples();
    writer_.close();
    std::cout << "\n[NRC] collected " << sampleCount_.load()
              << " / " << candidateCount_.load()
              << " (out_of_aabb_dropped=" << outOfAabbDroppedCount_.load() << ")"
              << " radiance GT records into " << datasetPath_ << std::endl;
}

Spectrum NRCCollectorIntegrator::Li(const Ray &initialRay, std::shared_ptr<Scene> scene) {
    const double eps = 1e-4;
    Spectrum L{0.0};
    Spectrum throughput{1.0};
    Ray ray = initialRay;
    int nBounces = 0;
    auto itsOpt = scene->intersect(ray);

    std::vector<BounceInfo> bounceInfos;

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

        double pSurvive = russianRoulette(throughput, nBounces);
        if (sampler->sample1D() >= pSurvive) {
            break;
        }
        throughput /= pSurvive;

        bounceInfos.emplace_back();
        BounceInfo &info = bounceInfos.back();
        info.position = its.position;
        info.wo = normalize(-ray.direction);
        info.normal = its.geometryNormal;
        info.bounce = static_cast<uint16_t>(nBounces);

        for (int i = 0; i < nDirectLightSamples; ++i) {
            auto sampleLightRecord = sampleDirectLighting(scene, its, ray);
            auto evalScatterRecord = evalScatter(its, ray, sampleLightRecord.wi);

            if (!sampleLightRecord.f.isBlack()) {
                double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                if (sampleLightRecord.isDelta) {
                    misw = 1.0;
                }

                Spectrum localValue = sampleLightRecord.f * evalScatterRecord.f
                                      / sampleLightRecord.pdf * misw / nDirectLightSamples;
                L += throughput * localValue;
                info.localRadiance += localValue / pSurvive;
            }
        }

        auto sampleScatterRecord = sampleScatter(its, ray);
        if (sampleScatterRecord.f.isBlack() || sampleScatterRecord.pdf == 0) {
            break;
        }

        info.bsdfWeight = sampleScatterRecord.f / sampleScatterRecord.pdf;
        info.hasBsdfWeight = true;
        throughput *= info.bsdfWeight;

        ray = Ray{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
        itsOpt = scene->intersect(ray);

        auto evalLightRecord = evalEmittance(scene, itsOpt, ray);
        if (!evalLightRecord.f.isBlack()) {
            double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
            if (sampleScatterRecord.isDelta) {
                misw = 1.0;
            }

            Spectrum bsdfHitLightValue = throughput * evalLightRecord.f * misw;
            L += bsdfHitLightValue;

            Spectrum localHitValue = info.bsdfWeight * evalLightRecord.f * misw / pSurvive;
            info.localRadiance += localHitValue;
        }
    }

    if (!bounceInfos.empty()) {
        for (int i = static_cast<int>(bounceInfos.size()) - 2; i >= 0; --i) {
            if (bounceInfos[i].hasBsdfWeight) {
                bounceInfos[i].localRadiance += bounceInfos[i + 1].localRadiance * bounceInfos[i].bsdfWeight;
            }
        }

        for (const auto &info : bounceInfos) {
            if (!info.localRadiance.isBlack() && !info.localRadiance.hasNaN()) {
                appendSample(info);
            }
        }
    }

    return L;
}

void NRCCollectorIntegrator::appendSample(const BounceInfo &info) {
    candidateCount_.fetch_add(1, std::memory_order_relaxed);

    bool needFlush = false;
    {
        std::lock_guard<std::mutex> lock(sampleMutex_);
        bumpBounceCount(candidateBounceCounts_, info.bounce);
        if (!insideFixedAabb(info.position)) {
            outOfAabbDroppedCount_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        const double pKeep = filterConfig_.keepProbability(info.bounce);
        const bool keep = pKeep >= 1.0 || (pKeep > 0.0 && rand_mt() < pKeep);
        if (!keep) {
            return;
        }

        RadianceSampleRecordV1 record{};
        record.position[0] = static_cast<float>(info.position.x);
        record.position[1] = static_cast<float>(info.position.y);
        record.position[2] = static_cast<float>(info.position.z);

        record.outgoingDir[0] = static_cast<float>(info.wo.x);
        record.outgoingDir[1] = static_cast<float>(info.wo.y);
        record.outgoingDir[2] = static_cast<float>(info.wo.z);

        record.normal[0] = static_cast<float>(info.normal.x);
        record.normal[1] = static_cast<float>(info.normal.y);
        record.normal[2] = static_cast<float>(info.normal.z);

        const RGB3 rgb = info.localRadiance.toRGB3();
        record.targetRGB[0] = static_cast<float>(rgb[0]);
        record.targetRGB[1] = static_cast<float>(rgb[1]);
        record.targetRGB[2] = static_cast<float>(rgb[2]);
        record.bounce = info.bounce;

        pendingSamples_.push_back(record);
        bumpBounceCount(bounceCounts_, record.bounce);
        needFlush = pendingSamples_.size() >= flushThreshold_;
    }
    sampleCount_.fetch_add(1, std::memory_order_relaxed);

    if (needFlush) {
        flushPendingSamples();
    }
}

void NRCCollectorIntegrator::flushPendingSamples() {
    std::vector<RadianceSampleRecordV1> localBatch;
    {
        std::lock_guard<std::mutex> lock(sampleMutex_);
        if (pendingSamples_.empty()) {
            return;
        }
        localBatch.swap(pendingSamples_);
    }
    writer_.append(localBatch.data(), localBatch.size());
}

}  // namespace NRC
