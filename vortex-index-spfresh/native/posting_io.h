// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "inc/Helper/DiskIO.h"

namespace vortex_spfresh {

// The factory's private, closed generation must remain immutable for this reader's lifetime.
class PostingFileIO : public SPTAG::Helper::DiskIO {
public:
    PostingFileIO() = default;
    PostingFileIO(const PostingFileIO &) = delete;
    PostingFileIO &operator=(const PostingFileIO &) = delete;

    ~PostingFileIO() override {
        ShutDown();
    }

    bool Initialize(const char *path, int mode, uint64_t = 1 << 20, uint32_t = 2, uint32_t = 2, uint16_t = 4)
        override {
        ShutDown();
        if (mode != (std::ios::binary | std::ios::in)) {
            return false;
        }
        const int fd = ::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
        if (fd < 0) {
            return false;
        }
        struct stat info {};
        const bool valid = ::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
                           uint64_t(info.st_size) <= uint64_t(std::numeric_limits<ptrdiff_t>::max());
        if (!valid) {
            ::close(fd);
            return false;
        }
        const auto size = static_cast<size_t>(info.st_size);
        void *mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            ::close(fd);
            return false;
        }
        fd_ = fd;
        mapping_ = mapping;
        size_ = size;
        return true;
    }

    const char *ReadBinaryView(uint64_t size, uint64_t offset) override {
        if (!mapping_ || offset > size_ || size > size_ - offset) {
            throw std::runtime_error("Posting view exceeds mapped file extent");
        }
        // Catch sequential truncation before dereferencing the map. Concurrent mutation
        // still violates the private generation's immutability contract.
        struct stat info {};
        if (::fstat(fd_, &info) != 0 || info.st_size < 0 || uint64_t(info.st_size) != size_) {
            throw std::runtime_error("Posting file size changed after open");
        }
        return static_cast<const char *>(mapping_) + offset;
    }

    uint64_t ReadBinary(uint64_t size, char *buffer, uint64_t offset = UINT64_MAX) override {
        if (offset == UINT64_MAX) {
            throw std::runtime_error("Posting reads require an explicit offset");
        }
        const char *view = PostingFileIO::ReadBinaryView(size, offset);
        std::memcpy(buffer, view, size);
        return size;
    }

    uint64_t WriteBinary(uint64_t, const char *, uint64_t = UINT64_MAX) override {
        throw std::runtime_error("Posting views are read-only");
    }

    uint64_t ReadString(uint64_t &, std::unique_ptr<char[]> &, char = '\n', uint64_t = UINT64_MAX) override {
        throw std::runtime_error("Posting views only support binary reads");
    }

    uint64_t WriteString(const char *, uint64_t = UINT64_MAX) override {
        throw std::runtime_error("Posting views are read-only");
    }

    uint64_t TellP() override {
        throw std::runtime_error("Posting views have no stream position");
    }

    void ShutDown() override {
        if (mapping_) {
            ::munmap(mapping_, size_);
            mapping_ = nullptr;
            size_ = 0;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    void *mapping_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
};

inline bool posting_views_enabled() {
    const char *value = std::getenv("VORTEX_SPFRESH_POSTING_VIEW");
    if (!value || std::strcmp(value, "1") == 0) {
        return true;
    }
    if (std::strcmp(value, "0") == 0) {
        return false;
    }
    throw std::runtime_error("VORTEX_SPFRESH_POSTING_VIEW must be 0 or 1");
}
} // namespace vortex_spfresh
