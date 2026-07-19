#include "NRC/NRCDatasetWriter.h"

#include <filesystem>

namespace NRC {

NRCDatasetWriter::~NRCDatasetWriter() {
    close();
}

bool NRCDatasetWriter::open(const std::string &path, const DatasetHeaderV1 &header) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (opened_) {
        return false;
    }

    const std::filesystem::path fsPath(path);
    if (fsPath.has_parent_path()) {
        std::filesystem::create_directories(fsPath.parent_path());
    }

    stream_.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!stream_.is_open()) {
        return false;
    }

    path_ = path;
    header_ = header;
    header_.recordCount = 0;
    recordCount_ = 0;

    stream_.write(reinterpret_cast<const char *>(&header_), sizeof(DatasetHeaderV1));
    if (!stream_.good()) {
        stream_.close();
        return false;
    }

    opened_ = true;
    return true;
}

bool NRCDatasetWriter::append(const RadianceSampleRecordV1 *records, size_t count) {
    if (count == 0) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_ || records == nullptr) {
        return false;
    }

    stream_.seekp(0, std::ios::end);
    stream_.write(reinterpret_cast<const char *>(records),
                  static_cast<std::streamsize>(sizeof(RadianceSampleRecordV1) * count));
    if (!stream_.good()) {
        return false;
    }

    recordCount_ += static_cast<uint64_t>(count);
    return true;
}

bool NRCDatasetWriter::flushHeaderLocked() {
    header_.recordCount = recordCount_;
    stream_.seekp(0, std::ios::beg);
    stream_.write(reinterpret_cast<const char *>(&header_), sizeof(DatasetHeaderV1));
    stream_.flush();
    return stream_.good();
}

bool NRCDatasetWriter::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_) {
        return true;
    }

    const bool headerOk = flushHeaderLocked();
    stream_.close();
    opened_ = false;
    return headerOk;
}

}  // namespace NRC
