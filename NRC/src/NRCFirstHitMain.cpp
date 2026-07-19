#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoreLayer/Adapter/JsonUtil.h"
#include "FunctionLayer/Camera/Camera.h"
#include "FunctionLayer/Camera/CameraFactory.h"
#include "FunctionLayer/Film/Film.h"
#include "FunctionLayer/Sampler/Independent.h"
#include "FunctionLayer/Scene/Scene.h"
#include "NRC/NRCBatchPredictor.h"
#include "ResourceLayer/File/FileUtils.h"

namespace {

struct FirstHitSettings {
    int spp = 4;
    std::string outputFile = "nrc_firsthit";

    std::filesystem::path testViewsFile;
    std::filesystem::path checkpointFile;
    std::filesystem::path trainMetaFile;
    std::filesystem::path inferScript = std::filesystem::path("NRC/python/infer_nrc_batch.py");
    std::string pythonCommand = "conda run -n moer-build python";
    size_t inferBatchSize = 1 << 20;
    float firstHitEnergyGain = 1.0f;
};

Json loadJsonFile(const std::filesystem::path &path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open json file: " + path.string());
    }
    Json json;
    in >> json;
    return json;
}

std::filesystem::path resolvePath(const std::filesystem::path &sceneDir, const std::filesystem::path &path) {
    if (path.empty()) {
        return path;
    }
    if (path.is_absolute()) {
        return path.lexically_normal();
    }
    return (sceneDir / path).lexically_normal();
}

std::vector<Json> loadViewTransforms(const Json &baseSceneJson,
                                     const std::filesystem::path &viewsPath) {
    std::vector<Json> transforms;
    if (baseSceneJson.contains("camera") && baseSceneJson["camera"].contains("transform")) {
        transforms.push_back(baseSceneJson["camera"]["transform"]);
    } else {
        throw std::runtime_error("scene.json missing camera.transform");
    }

    if (viewsPath.empty()) {
        return transforms;
    }

    const Json viewsJson = loadJsonFile(viewsPath);
    Json viewArray;
    if (viewsJson.is_array()) {
        viewArray = viewsJson;
    } else if (viewsJson.contains("views")) {
        viewArray = viewsJson["views"];
    } else if (viewsJson.contains("camera_transforms")) {
        viewArray = viewsJson["camera_transforms"];
    } else {
        throw std::runtime_error("Unsupported test views schema, expect array or {views:[...]}");
    }

    transforms.clear();
    for (const auto &entry : viewArray) {
        if (entry.contains("transform")) {
            transforms.push_back(entry["transform"]);
        } else {
            transforms.push_back(entry);
        }
    }
    if (transforms.empty()) {
        throw std::runtime_error("No test view transforms found in file: " + viewsPath.string());
    }
    return transforms;
}

FirstHitSettings parseSettings(const Json &sceneJson, const std::filesystem::path &sceneDir) {
    FirstHitSettings settings;
    if (!sceneJson.contains("renderer") || !sceneJson["renderer"].is_object()) {
        throw std::runtime_error("scene.json missing renderer object");
    }
    const Json &renderer = sceneJson["renderer"];
    settings.spp = getOptional(renderer, "spp", settings.spp);
    settings.outputFile = getOptional(renderer, "output_file", settings.outputFile);

    if (!renderer.contains("nrc") || !renderer["nrc"].is_object()) {
        throw std::runtime_error("renderer.nrc object is required");
    }
    const Json &nrc = renderer["nrc"];

    const std::string viewsFile = getOptional(nrc, "test_views_file", std::string());
    if (!viewsFile.empty()) {
        settings.testViewsFile = resolvePath(sceneDir, std::filesystem::path(viewsFile));
    }

    const std::string checkpoint = getOptional(nrc, "checkpoint_file", std::string());
    if (checkpoint.empty()) {
        throw std::runtime_error("renderer.nrc.checkpoint_file is required");
    }
    settings.checkpointFile = resolvePath(sceneDir, std::filesystem::path(checkpoint));

    const std::string trainMeta = getOptional(nrc, "train_meta_file", std::string());
    if (trainMeta.empty()) {
        settings.trainMetaFile = settings.checkpointFile.parent_path() / "train_meta.json";
    } else {
        settings.trainMetaFile = resolvePath(sceneDir, std::filesystem::path(trainMeta));
    }

    const std::string inferScript = getOptional(nrc, "infer_script", std::string());
    if (!inferScript.empty()) {
        settings.inferScript = resolvePath(sceneDir, std::filesystem::path(inferScript));
    } else {
        settings.inferScript = std::filesystem::absolute(settings.inferScript);
    }
    settings.pythonCommand = getOptional(nrc, "python_command", settings.pythonCommand);
    settings.inferBatchSize = static_cast<size_t>(getOptional(nrc, "infer_batch_size", static_cast<int>(settings.inferBatchSize)));
    settings.firstHitEnergyGain = getOptional(nrc, "firsthit_energy_gain", settings.firstHitEnergyGain);
    return settings;
}

void printProgressSimple(float percentage) {
    const int width = 50;
    const int done = static_cast<int>(percentage * width);
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < done ? '#' : '-');
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0f) << "%" << std::flush;
}

void renderOneViewFirstHit(const Json &baseSceneJson,
                           const Json &transform,
                           const std::filesystem::path &sceneDir,
                           const FirstHitSettings &settings,
                           int viewIndex) {
    Json sceneJson = baseSceneJson;
    sceneJson["camera"]["transform"] = transform;

    FileUtils::setWorkingDir(sceneDir.string() + "/");
    auto scene = std::make_shared<Scene>(sceneJson);
    scene->build();

    auto camera = CameraFactory::LoadCameraFromJson(sceneJson["camera"]);
    const Point2i resolution = getOptional(sceneJson["camera"], "resolution", Point2i(512, 512));
    const int width = resolution.x;
    const int height = resolution.y;
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

    auto film = std::make_unique<Film>(resolution, 3);
    auto sampler = std::make_shared<IndependentSampler>(settings.spp, 5);
    sampler->startPixel(Point2i(0, 0));
    auto localSampler = sampler->clone(0);

    NRC::NRCBatchPredictorConfig predictorCfg;
    predictorCfg.checkpointPath = settings.checkpointFile;
    predictorCfg.trainMetaPath = settings.trainMetaFile;
    predictorCfg.inferScriptPath = settings.inferScript;
    predictorCfg.pythonCommand = settings.pythonCommand;
    predictorCfg.inferBatchSize = settings.inferBatchSize;
    NRC::NRCBatchPredictor predictor(predictorCfg);

    std::vector<Spectrum> pixelSums(pixelCount, Spectrum(0.0));
    std::vector<uint32_t> pixelSamples(pixelCount, 0);
    std::vector<std::array<float, 6>> features;
    std::vector<size_t> featurePixelIds;
    features.reserve(pixelCount * static_cast<size_t>(settings.spp));
    featurePixelIds.reserve(pixelCount * static_cast<size_t>(settings.spp));

    const double eps = 1e-4;
    uint64_t finished = 0;
    const uint64_t total = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);

    std::cout << "[NRC] first-hit render view " << viewIndex
              << " (spp=" << settings.spp
              << ", firsthit_energy_gain=" << settings.firstHitEnergyGain << ")"
              << std::endl;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Point2i pixel(x, y);
            const size_t pixelId = static_cast<size_t>(y) * static_cast<size_t>(width)
                + static_cast<size_t>(x);
            localSampler->startPixel(pixel);

            for (int s = 0; s < settings.spp; ++s) {
                Ray ray = camera->generateRay(resolution, pixel, localSampler->getCameraSample());
                auto itsOpt = scene->intersect(ray);

                // Skip null materials and keep searching first valid surface hit.
                while (itsOpt.has_value() && itsOpt->material
                    && itsOpt->material->getBxDF(*itsOpt)->isNull()) {
                    ray = Ray{itsOpt->position + ray.direction * eps, ray.direction};
                    itsOpt = scene->intersect(ray);
                }

                if (itsOpt.has_value() && predictor.insideAabb(itsOpt->position)) {
                    features.push_back(predictor.buildFeature(itsOpt->position, normalize(-ray.direction)));
                    featurePixelIds.push_back(pixelId);
                }

                pixelSamples[pixelId] += 1;
                localSampler->nextSample();
            }
            ++finished;
        }
        printProgressSimple(static_cast<float>(finished) / static_cast<float>(total));
    }
    std::cout << std::endl;

    const auto tmpDir = sceneDir / "_nrc_tmp" / ("firsthit_view_" + std::to_string(viewIndex));
    auto preds = predictor.predict(features, tmpDir, "nrc_firsthit");
    if (preds.size() != featurePixelIds.size()) {
        throw std::runtime_error("NRC prediction size mismatch in first-hit render");
    }

    for (size_t i = 0; i < preds.size(); ++i) {
        const auto &p = preds[i];
        const RGB3 rgb(
            std::max(0.0, static_cast<double>(p[0])),
            std::max(0.0, static_cast<double>(p[1])),
            std::max(0.0, static_cast<double>(p[2])));
        pixelSums[featurePixelIds[i]] += Spectrum(rgb) * static_cast<double>(settings.firstHitEnergyGain);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t id = static_cast<size_t>(y) * static_cast<size_t>(width)
                + static_cast<size_t>(x);
            const uint32_t n = pixelSamples[id];
            if (n == 0) {
                film->deposit(Point2i(x, y), Spectrum(0.0));
            } else {
                film->deposit(Point2i(x, y), pixelSums[id] / static_cast<double>(n));
            }
        }
    }

    const std::string outputPath = settings.outputFile + "_nrc_firsthit_view_" + std::to_string(viewIndex);
    film->save(outputPath);
    std::cout << "[NRC] saved first-hit image -> " << outputPath << std::endl;
}

}  // namespace

int main(int argc, const char *argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: Moer-NRC-FirstHit <scene_dir>" << std::endl;
            return 1;
        }

        Spectrum::init();
        const std::filesystem::path sceneDir = std::filesystem::path(argv[1]);
        const std::filesystem::path scenePath = sceneDir / "scene.json";
        const Json baseSceneJson = loadJsonFile(scenePath);
        const FirstHitSettings settings = parseSettings(baseSceneJson, sceneDir);
        const auto transforms = loadViewTransforms(baseSceneJson, settings.testViewsFile);

        std::cout << "[NRC] scene: " << scenePath.string() << std::endl;
        std::cout << "[NRC] views: " << transforms.size() << std::endl;

        for (size_t i = 0; i < transforms.size(); ++i) {
            renderOneViewFirstHit(
                baseSceneJson, transforms[i], sceneDir, settings, static_cast<int>(i));
        }

        std::cout << "[NRC] first-hit render finished." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[NRC] fatal: " << e.what() << std::endl;
        return 2;
    }
}
