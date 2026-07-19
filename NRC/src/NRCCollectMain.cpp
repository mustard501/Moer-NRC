#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "CoreLayer/Adapter/JsonUtil.h"
#include "FunctionLayer/Camera/CameraFactory.h"
#include "FunctionLayer/Film/Film.h"
#include "FunctionLayer/Integrator/PathIntegrator-new.h"
#include "FunctionLayer/Sampler/Independent.h"
#include "FunctionLayer/Scene/Scene.h"
#include "FunctionLayer/TileGenerator/SequenceTileGenerator.h"
#include "NRC/NRCCollectorIntegrator.h"
#include "NRC/NRCTypes.h"
#include "ResourceLayer/File/FileUtils.h"

namespace {

struct CollectorSettings {
    int spp = 32;
    int threads = 12;
    size_t flushThreshold = 1 << 15;
    std::string datasetDir = "NRC/runtime/";
    std::string datasetPrefix = "scene";
    std::string viewsFile;
    NRC::SampleFilterConfig sampleFilter;
};

struct ViewBounceStats {
    int viewIndex = 0;
    std::string binFile;
    uint64_t writtenTotal = 0;
    uint64_t candidateTotal = 0;
    uint64_t outOfAabbDropped = 0;
    std::vector<uint64_t> bounceCounts;
    std::vector<uint64_t> candidateBounceCounts;
};

struct FixedAabb {
    std::array<float, 3> min = {0.f, 0.f, 0.f};
    std::array<float, 3> max = {0.f, 0.f, 0.f};
};

struct MeshInstance {
    std::filesystem::path objPath;
    std::array<double, 3> position = {0.0, 0.0, 0.0};
    std::array<double, 3> scale = {1.0, 1.0, 1.0};
    std::string sourceLabel;
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

std::array<double, 3> parseVec3OrScalar(const Json &value, const std::array<double, 3> &fallback) {
    if (value.is_number()) {
        const double s = value.get<double>();
        return {s, s, s};
    }
    if (value.is_array() && value.size() >= 3) {
        return {value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
    }
    return fallback;
}

std::vector<MeshInstance> collectMeshInstances(const Json &sceneJson, const std::filesystem::path &sceneDir) {
    std::vector<MeshInstance> meshes;
    if (!sceneJson.contains("entities") || !sceneJson["entities"].is_array()) {
        throw std::runtime_error("scene.json missing entities array");
    }

    for (const auto &entity : sceneJson["entities"]) {
        if (!entity.is_object()) {
            continue;
        }
        if (!entity.contains("type") || entity["type"] != "mesh") {
            continue;
        }
        if (!entity.contains("file")) {
            continue;
        }

        MeshInstance inst;
        std::filesystem::path path = entity["file"].get<std::string>();
        if (!path.is_absolute()) {
            path = sceneDir / path;
        }
        inst.objPath = path.lexically_normal();
        const std::string material = entity.contains("material")
            ? entity["material"].get<std::string>()
            : std::string("<none>");
        inst.sourceLabel =
            inst.objPath.filename().string() + " material=" + material;

        if (entity.contains("transform") && entity["transform"].is_object()) {
            const Json &t = entity["transform"];
            if (t.contains("position")) {
                inst.position = parseVec3OrScalar(t["position"], inst.position);
            }
            if (t.contains("scale")) {
                inst.scale = parseVec3OrScalar(t["scale"], inst.scale);
            }
            if (t.contains("rotation")) {
                std::cerr << "[NRC] warning: mesh transform.rotation is ignored for OBJ AABB pre-pass."
                          << std::endl;
            }
        }

        meshes.push_back(inst);
    }

    if (meshes.empty()) {
        throw std::runtime_error("No mesh entities with OBJ file found in scene.json");
    }
    return meshes;
}

std::vector<std::array<double, 3>> loadObjVertices(const std::filesystem::path &objPath) {
    std::ifstream in(objPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open OBJ: " + objPath.string());
    }

    std::vector<std::array<double, 3>> vertices;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 3) {
            continue;
        }
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream iss(line.substr(2));
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (iss >> x >> y >> z) {
                vertices.push_back({x, y, z});
            }
        }
    }
    if (vertices.empty()) {
        throw std::runtime_error("OBJ has no vertex records: " + objPath.string());
    }
    return vertices;
}

FixedAabb computeObjAabbFromScene(const Json &sceneJson, const std::filesystem::path &sceneDir) {
    const auto meshes = collectMeshInstances(sceneJson, sceneDir);

    std::unordered_map<std::string, std::vector<std::array<double, 3>>> vertexCache;
    std::array<double, 3> minv = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maxv = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    std::array<std::string, 3> minSource = {"<unknown>", "<unknown>", "<unknown>"};
    std::array<std::string, 3> maxSource = {"<unknown>", "<unknown>", "<unknown>"};

    uint64_t transformedVertices = 0;
    for (const auto &mesh : meshes) {
        const std::string key = mesh.objPath.string();
        auto it = vertexCache.find(key);
        if (it == vertexCache.end()) {
            it = vertexCache.emplace(key, loadObjVertices(mesh.objPath)).first;
        }
        const auto &verts = it->second;

        for (const auto &v : verts) {
            const std::array<double, 3> world = {
                v[0] * mesh.scale[0] + mesh.position[0],
                v[1] * mesh.scale[1] + mesh.position[1],
                v[2] * mesh.scale[2] + mesh.position[2]};

            minv[0] = std::min(minv[0], world[0]);
            minv[1] = std::min(minv[1], world[1]);
            minv[2] = std::min(minv[2], world[2]);
            maxv[0] = std::max(maxv[0], world[0]);
            maxv[1] = std::max(maxv[1], world[1]);
            maxv[2] = std::max(maxv[2], world[2]);
            if (world[0] == minv[0]) {
                minSource[0] = mesh.sourceLabel;
            }
            if (world[1] == minv[1]) {
                minSource[1] = mesh.sourceLabel;
            }
            if (world[2] == minv[2]) {
                minSource[2] = mesh.sourceLabel;
            }
            if (world[0] == maxv[0]) {
                maxSource[0] = mesh.sourceLabel;
            }
            if (world[1] == maxv[1]) {
                maxSource[1] = mesh.sourceLabel;
            }
            if (world[2] == maxv[2]) {
                maxSource[2] = mesh.sourceLabel;
            }
        }
        transformedVertices += static_cast<uint64_t>(verts.size());
    }

    if (!std::isfinite(minv[0]) || !std::isfinite(maxv[0])) {
        throw std::runtime_error("Failed to compute finite OBJ AABB from scene meshes");
    }

    FixedAabb out;
    out.min = {
        static_cast<float>(minv[0]),
        static_cast<float>(minv[1]),
        static_cast<float>(minv[2])};
    out.max = {
        static_cast<float>(maxv[0]),
        static_cast<float>(maxv[1]),
        static_cast<float>(maxv[2])};

    std::cout << "[NRC] mesh OBJ pre-pass: instances=" << meshes.size()
              << ", unique_obj_files=" << vertexCache.size()
              << ", transformed_vertices=" << transformedVertices << std::endl;
    std::cout << "[NRC] fixed AABB(min) = [" << out.min[0] << ", " << out.min[1] << ", " << out.min[2] << "]"
              << std::endl;
    std::cout << "[NRC] fixed AABB(max) = [" << out.max[0] << ", " << out.max[1] << ", " << out.max[2] << "]"
              << std::endl;
    std::cout << "[NRC] AABB source min.x -> " << minSource[0] << std::endl;
    std::cout << "[NRC] AABB source min.y -> " << minSource[1] << std::endl;
    std::cout << "[NRC] AABB source min.z -> " << minSource[2] << std::endl;
    std::cout << "[NRC] AABB source max.x -> " << maxSource[0] << std::endl;
    std::cout << "[NRC] AABB source max.y -> " << maxSource[1] << std::endl;
    std::cout << "[NRC] AABB source max.z -> " << maxSource[2] << std::endl;

    return out;
}

std::vector<Json> loadViewTransforms(const Json &baseSceneJson, const std::filesystem::path &sceneDir, const std::string &viewsFile) {
    std::vector<Json> transforms;
    if (baseSceneJson.contains("camera") && baseSceneJson["camera"].contains("transform")) {
        transforms.push_back(baseSceneJson["camera"]["transform"]);
    } else {
        throw std::runtime_error("scene.json missing camera.transform");
    }

    if (viewsFile.empty()) {
        return transforms;
    }

    std::filesystem::path viewsPath = std::filesystem::path(viewsFile);
    if (!viewsPath.is_absolute()) {
        viewsPath = sceneDir / viewsPath;
    }
    Json viewsJson = loadJsonFile(viewsPath);

    Json viewArray;
    if (viewsJson.is_array()) {
        viewArray = viewsJson;
    } else if (viewsJson.contains("views")) {
        viewArray = viewsJson["views"];
    } else if (viewsJson.contains("camera_transforms")) {
        viewArray = viewsJson["camera_transforms"];
    } else {
        throw std::runtime_error("Unsupported views file schema, expect array or {views:[...]}");
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
        throw std::runtime_error("No view transforms found in views file");
    }
    return transforms;
}

CollectorSettings parseSettings(const Json &sceneJson, const std::filesystem::path &sceneDir) {
    CollectorSettings settings;
    if (sceneJson.contains("renderer")) {
        const Json &renderer = sceneJson["renderer"];
        settings.spp = getOptional(renderer, "spp", settings.spp);

        if (renderer.contains("nrc")) {
            const Json &nrc = renderer["nrc"];
            settings.datasetDir = getOptional(nrc, "dataset_dir", settings.datasetDir);
            settings.datasetPrefix = getOptional(nrc, "dataset_prefix", sceneDir.filename().string());
            settings.viewsFile = getOptional(nrc, "views_file", std::string());
            settings.threads = getOptional(nrc, "threads", settings.threads);
            settings.flushThreshold = static_cast<size_t>(getOptional(nrc, "flush_threshold", static_cast<int>(settings.flushThreshold)));

            settings.sampleFilter.maxBounceRecord =
                getOptional(nrc, "max_bounce_record", settings.sampleFilter.maxBounceRecord);
            settings.sampleFilter.globalKeep =
                getOptional(nrc, "global_keep", settings.sampleFilter.globalKeep);
            if (nrc.contains("bounce_keep_probs") && nrc["bounce_keep_probs"].is_array()) {
                settings.sampleFilter.bounceKeepProbs.clear();
                for (const auto &value : nrc["bounce_keep_probs"]) {
                    settings.sampleFilter.bounceKeepProbs.push_back(value.get<double>());
                }
            }
        }
    }

    if (settings.datasetPrefix.empty()) {
        settings.datasetPrefix = sceneDir.filename().string();
    }
    if (settings.sampleFilter.maxBounceRecord < 0) {
        throw std::runtime_error("renderer.nrc.max_bounce_record must be >= 0");
    }
    if (settings.sampleFilter.globalKeep < 0.0) {
        throw std::runtime_error("renderer.nrc.global_keep must be >= 0");
    }
    for (double p : settings.sampleFilter.bounceKeepProbs) {
        if (p < 0.0) {
            throw std::runtime_error("renderer.nrc.bounce_keep_probs entries must be >= 0");
        }
    }
    return settings;
}

void printSampleFilter(const NRC::SampleFilterConfig &filter) {
    std::cout << "[NRC] sample filter: max_bounce_record=" << filter.maxBounceRecord
              << ", global_keep=" << filter.globalKeep
              << ", bounce_keep_probs=[";
    for (size_t i = 0; i < filter.bounceKeepProbs.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << "b" << (i + 1) << "=" << filter.bounceKeepProbs[i];
    }
    std::cout << "]" << std::endl;
}

void writeBounceStatsFile(const std::filesystem::path &statsPath,
                          const std::vector<ViewBounceStats> &allStats,
                          const NRC::SampleFilterConfig &filter) {
    if (statsPath.has_parent_path()) {
        std::filesystem::create_directories(statsPath.parent_path());
    }

    std::ofstream out(statsPath, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open bounce stats file: " + statsPath.string());
    }

    out << "# NRC bounce sample counts (after write-time keep/drop)\n";
    out << "# written = kept samples, candidates = before keep/drop\n";
    out << "# out_of_aabb_dropped = candidates rejected by fixed mesh OBJ AABB\n";
    out << "# max_bounce_record=" << filter.maxBounceRecord
        << " global_keep=" << filter.globalKeep << "\n";
    out << "# bounce_keep_probs=";
    for (size_t i = 0; i < filter.bounceKeepProbs.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << filter.bounceKeepProbs[i];
    }
    out << "\n\n";

    uint64_t grandWritten = 0;
    uint64_t grandCandidates = 0;
    for (const auto &view : allStats) {
        out << "[view " << view.viewIndex << "] file=" << view.binFile
            << " written=" << view.writtenTotal
            << " candidates=" << view.candidateTotal
            << " out_of_aabb_dropped=" << view.outOfAabbDropped << "\n";

        const size_t maxBounce = std::max(view.bounceCounts.size(), view.candidateBounceCounts.size());
        for (size_t bounce = 1; bounce < maxBounce; ++bounce) {
            const uint64_t written =
                bounce < view.bounceCounts.size() ? view.bounceCounts[bounce] : 0;
            const uint64_t candidates =
                bounce < view.candidateBounceCounts.size() ? view.candidateBounceCounts[bounce] : 0;
            if (written == 0 && candidates == 0) {
                continue;
            }
            out << "  bounce " << bounce << ": written=" << written
                << " candidates=" << candidates << "\n";
        }
        out << "\n";
        grandWritten += view.writtenTotal;
        grandCandidates += view.candidateTotal;
    }

    out << "[summary] views=" << allStats.size()
        << " written_total=" << grandWritten
        << " candidates_total=" << grandCandidates << "\n";
    out.flush();
    if (!out.good()) {
        throw std::runtime_error("Failed to write bounce stats file: " + statsPath.string());
    }
}

ViewBounceStats runCollectionForView(const Json &baseSceneJson,
                                     const Json &transform,
                                     const std::filesystem::path &sceneDir,
                                     const std::filesystem::path &datasetPath,
                                     const CollectorSettings &settings,
                                     const FixedAabb &fixedAabb,
                                     int viewIndex) {
    Json sceneJson = baseSceneJson;
    sceneJson["camera"]["transform"] = transform;

    FileUtils::setWorkingDir(sceneDir.string() + "/");

    std::shared_ptr<Scene> scene = std::make_shared<Scene>(sceneJson);
    scene->build();

    auto camera = CameraFactory::LoadCameraFromJson(sceneJson["camera"]);
    Point2i resolution = getOptional(sceneJson["camera"], "resolution", Point2i(512, 512));

    std::cout << "[NRC] collecting view " << viewIndex
              << " -> " << datasetPath.string() << std::endl;

    NRC::NRCCollectorIntegrator collector(
        camera,
        std::make_unique<Film>(resolution, 3),
        std::make_unique<SequenceTileGenerator>(resolution),
        std::make_shared<IndependentSampler>(settings.spp, 5),
        settings.spp,
        datasetPath.string(),
        settings.threads,
        settings.flushThreshold,
        settings.sampleFilter,
        true,
        fixedAabb.min,
        fixedAabb.max);
    collector.render(scene);

    ViewBounceStats stats;
    stats.viewIndex = viewIndex;
    stats.binFile = datasetPath.filename().string();
    stats.writtenTotal = collector.collectedSamples();
    stats.candidateTotal = collector.candidateSamples();
    stats.outOfAabbDropped = collector.outOfAabbDroppedSamples();
    stats.bounceCounts = collector.bounceCounts();
    stats.candidateBounceCounts = collector.candidateBounceCounts();

    std::cout << "[NRC] view " << viewIndex
              << " kept " << stats.writtenTotal << " / " << stats.candidateTotal
              << " (out_of_aabb_dropped=" << stats.outOfAabbDropped << ")"
              << " bounce distribution:";
    for (size_t bounce = 1; bounce < stats.bounceCounts.size(); ++bounce) {
        if (stats.bounceCounts[bounce] == 0) {
            continue;
        }
        std::cout << " b" << bounce << "=" << stats.bounceCounts[bounce];
    }
    std::cout << std::endl;

    return stats;
}

}  // namespace

int main(int argc, const char *argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: Moer-NRC-Collect <scene_dir>\n";
            std::cerr << "Reads <scene_dir>/scene.json and optional renderer.nrc.views_file." << std::endl;
            return 1;
        }

        Spectrum::init();

        const std::filesystem::path sceneDir = std::filesystem::path(argv[1]);
        const std::filesystem::path scenePath = sceneDir / "scene.json";
        Json baseSceneJson = loadJsonFile(scenePath);
        CollectorSettings settings = parseSettings(baseSceneJson, sceneDir);
        std::vector<Json> transforms = loadViewTransforms(baseSceneJson, sceneDir, settings.viewsFile);
        const FixedAabb fixedAabb = computeObjAabbFromScene(baseSceneJson, sceneDir);

        std::filesystem::path datasetDir(settings.datasetDir);
        if (!datasetDir.is_absolute()) {
            datasetDir = std::filesystem::current_path() / datasetDir;
        }
        std::filesystem::create_directories(datasetDir);

        const std::filesystem::path viewDatasetDir = (datasetDir / sceneDir).lexically_normal();
        std::filesystem::create_directories(viewDatasetDir);
        const std::filesystem::path bounceStatsPath =
            viewDatasetDir / (settings.datasetPrefix + "_bounce_stats.txt");

        std::cout << "[NRC] scene: " << scenePath.string() << std::endl;
        std::cout << "[NRC] views: " << transforms.size() << ", spp: " << settings.spp
                  << ", threads: " << settings.threads << std::endl;
        printSampleFilter(settings.sampleFilter);
        std::cout << "[NRC] bounce stats -> " << bounceStatsPath.string() << std::endl;

        std::vector<ViewBounceStats> allBounceStats;
        allBounceStats.reserve(transforms.size());

        for (size_t i = 0; i < transforms.size(); ++i) {
            auto datasetPath = viewDatasetDir / (settings.datasetPrefix + "_view_" + std::to_string(i) + ".bin");
            allBounceStats.push_back(
                runCollectionForView(baseSceneJson, transforms[i], sceneDir, datasetPath, settings, fixedAabb, static_cast<int>(i)));
            writeBounceStatsFile(bounceStatsPath, allBounceStats, settings.sampleFilter);
        }

        std::cout << "[NRC] collection finished." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[NRC] fatal: " << e.what() << std::endl;
        return 2;
    }
}
