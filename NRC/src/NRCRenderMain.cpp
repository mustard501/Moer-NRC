#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoreLayer/Adapter/JsonUtil.h"
#include "FunctionLayer/Camera/CameraFactory.h"
#include "FunctionLayer/Film/Film.h"
#include "FunctionLayer/Sampler/Independent.h"
#include "FunctionLayer/Scene/Scene.h"
#include "NRC/NRCBatchPredictor.h"
#include "NRC/NRCPathCutoffIntegrator.h"
#include "ResourceLayer/File/FileUtils.h"

namespace {

struct NRCRenderSettings {
    int spp = 4;
    std::string outputFile = "nrc_render";

    std::filesystem::path testViewsFile;
    std::filesystem::path checkpointFile;
    std::filesystem::path trainMetaFile;
    std::filesystem::path inferScript = std::filesystem::path("NRC/python/infer_nrc_batch.py");
    std::string pythonCommand = "conda run -n moer-build python";
    int ptMaxBounce = 4;
    size_t inferBatchSize = 1 << 20;
    float energyGain = 1.0f;
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
                                     const std::filesystem::path &sceneDir,
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

NRCRenderSettings parseSettings(const Json &sceneJson, const std::filesystem::path &sceneDir) {
    NRCRenderSettings settings;
    if (!sceneJson.contains("renderer") || !sceneJson["renderer"].is_object()) {
        throw std::runtime_error("scene.json missing renderer object");
    }
    const Json &renderer = sceneJson["renderer"];
    settings.spp = getOptional(renderer, "spp", settings.spp);
    settings.outputFile = getOptional(renderer, "output_file", settings.outputFile);

    if (!renderer.contains("nrc") || !renderer["nrc"].is_object()) {
        throw std::runtime_error("renderer.nrc object is required for NRC render mode");
    }
    const Json &nrc = renderer["nrc"];

    const std::string viewsFile = getOptional(nrc, "test_views_file", std::string());
    if (viewsFile.empty()) {
        throw std::runtime_error("renderer.nrc.test_views_file is required");
    }
    settings.testViewsFile = resolvePath(sceneDir, std::filesystem::path(viewsFile));

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
    settings.ptMaxBounce = getOptional(nrc, "pt_max_bounce", settings.ptMaxBounce);
    settings.inferBatchSize = static_cast<size_t>(getOptional(nrc, "infer_batch_size", static_cast<int>(settings.inferBatchSize)));
    settings.energyGain = getOptional(nrc, "energy_gain", settings.energyGain);
    return settings;
}

void renderOneView(const Json &baseSceneJson,
                   const Json &transform,
                   const std::filesystem::path &sceneDir,
                   const NRCRenderSettings &settings,
                   int viewIndex) {
    Json sceneJson = baseSceneJson;
    sceneJson["camera"]["transform"] = transform;

    FileUtils::setWorkingDir(sceneDir.string() + "/");

    std::shared_ptr<Scene> scene = std::make_shared<Scene>(sceneJson);
    scene->build();

    auto camera = CameraFactory::LoadCameraFromJson(sceneJson["camera"]);
    const Point2i resolution = getOptional(sceneJson["camera"], "resolution", Point2i(512, 512));

    NRC::NRCBatchPredictorConfig predictorCfg;
    predictorCfg.checkpointPath = settings.checkpointFile;
    predictorCfg.trainMetaPath = settings.trainMetaFile;
    predictorCfg.inferScriptPath = settings.inferScript;
    predictorCfg.pythonCommand = settings.pythonCommand;
    predictorCfg.inferBatchSize = settings.inferBatchSize;

    const auto tempDir = sceneDir / "_nrc_tmp" / ("view_" + std::to_string(viewIndex));
    NRC::NRCPathCutoffIntegrator integrator(
        camera,
        std::make_unique<Film>(resolution, 3),
        std::make_shared<IndependentSampler>(settings.spp, 5),
        settings.spp,
        settings.ptMaxBounce,
        NRC::NRCBatchPredictor(predictorCfg),
        tempDir,
        settings.energyGain);

    std::cout << "[NRC] rendering test view " << viewIndex
              << " (spp=" << settings.spp
              << ", pt_max_bounce=" << settings.ptMaxBounce
              << ", energy_gain=" << settings.energyGain << ")"
              << std::endl;
    integrator.render(scene);
    const std::string outputPath = settings.outputFile + "_nrc_view_" + std::to_string(viewIndex);
    integrator.save(outputPath);
    std::cout << "\n[NRC] saved view " << viewIndex << " -> " << outputPath << std::endl;
}

}  // namespace

int main(int argc, const char *argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: Moer-NRC-Render <scene_dir>" << std::endl;
            std::cerr << "Reads <scene_dir>/scene.json and renderer.nrc.test_views_file." << std::endl;
            return 1;
        }

        Spectrum::init();

        const std::filesystem::path sceneDir = std::filesystem::path(argv[1]);
        const std::filesystem::path scenePath = sceneDir / "scene.json";
        const Json baseSceneJson = loadJsonFile(scenePath);
        const NRCRenderSettings settings = parseSettings(baseSceneJson, sceneDir);
        const auto transforms = loadViewTransforms(baseSceneJson, sceneDir, settings.testViewsFile);

        std::cout << "[NRC] scene: " << scenePath.string() << std::endl;
        std::cout << "[NRC] test views file: " << settings.testViewsFile.string()
                  << " (" << transforms.size() << " views)" << std::endl;
        std::cout << "[NRC] checkpoint: " << settings.checkpointFile.string() << std::endl;
        std::cout << "[NRC] train meta: " << settings.trainMetaFile.string() << std::endl;

        for (size_t i = 0; i < transforms.size(); ++i) {
            renderOneView(baseSceneJson, transforms[i], sceneDir, settings, static_cast<int>(i));
        }

        std::cout << "[NRC] render finished." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[NRC] fatal: " << e.what() << std::endl;
        return 2;
    }
}
