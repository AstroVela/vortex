// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vortex_rabitq {
constexpr size_t io_alignment = 4096;
constexpr size_t rerank_batch = 512;
constexpr size_t max_request_bytes = 128 * 1024;

inline std::runtime_error io_error(const char *operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

class OriginalFile {
public:
    int fd = -1;
    uint64_t length = 0;
    explicit OriginalFile(const std::filesystem::path &path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        if (fd < 0) {
            throw io_error("Cannot open original Float32 file");
        }
        try {
            struct stat info {};
            if (::fstat(fd, &info)) {
                throw io_error("Cannot stat original Float32 file");
            }
            if (!S_ISREG(info.st_mode) || info.st_size <= 0 ||
                uint64_t(info.st_size) > uint64_t(std::numeric_limits<ptrdiff_t>::max())) {
                throw std::runtime_error("Invalid original Float32 file");
            }
            length = uint64_t(info.st_size);
        } catch (...) {
            ::close(fd);
            fd = -1;
            throw;
        }
    }
    OriginalFile(const OriginalFile &) = delete;
    OriginalFile &operator=(const OriginalFile &) = delete;
    ~OriginalFile() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    void check_size() const {
        struct stat info {};
        if (::fstat(fd, &info)) {
            throw io_error("Cannot stat original Float32 file");
        }
        if (info.st_size < 0 || uint64_t(info.st_size) != length) {
            throw std::runtime_error("Original Float32 file size changed");
        }
    }
};

struct IoStats {
    uint64_t requests = 0, bytes = 0;
};

class OriginalReader {
    struct Range {
        uint32_t file;
        uint64_t start, end;
        size_t buffer_offset = 0;
    };
    struct Reference {
        size_t request;
        uint64_t offset;
    };
    struct Free {
        void operator()(uint8_t *p) const {
            std::free(p);
        }
    };
    std::array<std::unique_ptr<OriginalFile>, 2> files_;
    std::unique_ptr<uint8_t, Free> buffer_;
    size_t capacity_ = 0;
    std::vector<Range> ranges_;
    std::vector<Reference> references_;

public:
    explicit OriginalReader(const std::filesystem::path &root) {
        files_[0] = std::make_unique<OriginalFile>(root / "postings.bin");
        files_[1] = std::make_unique<OriginalFile>(root / "vectors.bin");
    }
    uint64_t length(uint32_t file) const {
        return files_.at(file)->length;
    }
    void prepare(const std::vector<uint64_t> &positions, size_t vector_bytes, IoStats &stats) {
        if (positions.size() > rerank_batch || !vector_bytes || vector_bytes > 4096 * sizeof(float)) {
            throw std::runtime_error("Invalid original-vector read batch");
        }
        for (const auto &file : files_) {
            file->check_size();
        }
        struct Input {
            uint32_t file;
            uint64_t offset;
            size_t index;
        };
        std::vector<Input> inputs;
        for (size_t i = 0; i < positions.size(); ++i) {
            const auto file = uint32_t(positions[i] >> 63);
            const auto offset = positions[i] & ~(uint64_t(1) << 63);
            if (offset % sizeof(float) || offset > files_[file]->length ||
                vector_bytes > files_[file]->length - offset) {
                throw std::runtime_error("Invalid original-vector read extent");
            }
            inputs.push_back({file, offset, i});
        }
        std::sort(inputs.begin(), inputs.end(), [](const Input &a, const Input &b) {
            return a.file != b.file ? a.file < b.file : a.offset < b.offset;
        });
        ranges_.clear();
        references_.resize(positions.size());
        for (auto input : inputs) {
            const auto start = input.offset / io_alignment * io_alignment;
            const auto end = (input.offset + vector_bytes + io_alignment - 1) / io_alignment * io_alignment;
            if (!ranges_.empty() && ranges_.back().file == input.file && start <= ranges_.back().end &&
                end - ranges_.back().start <= max_request_bytes) {
                ranges_.back().end = std::max(ranges_.back().end, end);
            } else {
                ranges_.push_back({input.file, start, end});
            }
            references_[input.index] = {ranges_.size() - 1, input.offset - ranges_.back().start};
        }
        size_t bytes = 0;
        for (auto &range : ranges_) {
            range.buffer_offset = bytes;
            bytes += size_t(range.end - range.start);
        }
        if (bytes > capacity_) {
            void *allocation = nullptr;
            if (::posix_memalign(&allocation, io_alignment, bytes)) {
                throw std::runtime_error("Cannot allocate aligned original-vector IO buffer");
            }
            buffer_.reset(static_cast<uint8_t *>(allocation));
            capacity_ = bytes;
        }
        stats.requests += ranges_.size();
        stats.bytes += bytes;
        for (const auto &range : ranges_) {
            const auto needed = std::min(range.end, files_[range.file]->length) - range.start;
            uint64_t done = 0;
            while (done < needed) {
                const auto result = ::pread(files_[range.file]->fd,
                                            buffer_.get() + range.buffer_offset + done,
                                            size_t(needed - done),
                                            off_t(range.start + done));
                if (result < 0 && errno == EINTR) {
                    continue;
                }
                if (result < 0) {
                    throw io_error("Buffered original-vector pread failed");
                }
                if (!result) {
                    throw std::runtime_error("Short buffered original-vector read");
                }
                done += uint64_t(result);
            }
        }
    }
    void copy(size_t index, float *destination, size_t bytes) const {
        const auto reference = references_.at(index);
        const auto *source = buffer_.get() + ranges_[reference.request].buffer_offset + reference.offset;
        std::memcpy(destination, source, bytes);
    }
};
} // namespace vortex_rabitq
