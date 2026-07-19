#pragma once

#include <cstddef>
#include <fstream>
#include <mutex>
#include <string>
#include "NRC/NRCTypes.h"

namespace NRC {

class NRCDatasetWriter {
public:
    NRCDatasetWriter() = default;
    ~NRCDatasetWriter();

    bool open(const std::string &path, const DatasetHeaderV1 &header);
    bool append(const RadianceSampleRecordV1 *records, size_t count);
    bool close();

    [[nodiscard]]
    uint64_t recordCount() const { return recordCount_; }

private:
    bool flushHeaderLocked();

    std::mutex mutex_;
    std::string path_;
    DatasetHeaderV1 header_{};
    std::fstream stream_;
    uint64_t recordCount_ = 0;
    bool opened_ = false;
};

}  // namespace NRC
