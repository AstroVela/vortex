// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "bridge.h"
#include "posting_io.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"

namespace {
void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class F>
void rejects(F &&function, const char *message) {
    try {
        function();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error(message);
}

struct Scratch {
    std::filesystem::path path;
    Scratch() {
        const auto pattern = (std::filesystem::temp_directory_path() / "spfresh-posting-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        require(mkdtemp(name.data()) != nullptr, "Cannot create private test scratch");
        path = name.data();
    }
    ~Scratch() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void write_bytes(const std::filesystem::path &path, const std::string &bytes) {
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(bytes.data(), bytes.size());
}

std::string read_bytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Cannot read test file");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void test_reader(const std::filesystem::path &root) {
    const auto path = root / "bytes.bin";
    const auto next = root / "next.bin";
    const std::string bytes("posting\0bytes\n", 14);
    write_bytes(path, bytes);
    write_bytes(next, "next");
    vortex_spfresh::PostingFileIO reader;
    require(reader.Initialize(path.c_str(), std::ios::binary | std::ios::in), "Cannot map test file");
    const char *view = reader.ReadBinaryView(bytes.size(), 0);
    require(std::string(view, bytes.size()) == bytes, "Mapped bytes changed");
    require(reader.ReadBinaryView(4, 2) == view + 2, "View copied bytes instead of borrowing");
    require(reader.ReadBinaryView(0, bytes.size()) == view + bytes.size(), "Empty boundary view failed");
    for (const auto [size, offset] : std::array<std::pair<uint64_t, uint64_t>, 4> {
             {{1, bytes.size()}, {0, bytes.size() + 1}, {UINT64_MAX, 1}, {1, UINT64_MAX}}}) {
        rejects([&] { reader.ReadBinaryView(size, offset); }, "Out-of-bounds view was accepted");
    }
    std::array<char, 4> copy {};
    require(reader.ReadBinary(copy.size(), copy.data(), 2) == copy.size() &&
                std::string(copy.data(), copy.size()) == bytes.substr(2, 4),
            "Copying fallback differs from borrowed bytes");
    rejects([&] { reader.ReadBinary(1, copy.data(), bytes.size()); }, "Copying read exceeded file");
    rejects([&] { reader.ReadBinary(1, copy.data()); }, "Implicit stream position was accepted");
    rejects([&] { reader.WriteBinary(1, "x", 0); }, "Read-only mapping accepted writes");
    std::filesystem::remove(path);
    write_bytes(path, "replacement");
    require(std::string(view, bytes.size()) == bytes, "Path replacement invalidated an owned view");
    require(reader.ReadBinary(copy.size(), copy.data(), 2) == copy.size() &&
                std::string(copy.data(), copy.size()) == bytes.substr(2, 4),
            "Copying fallback reopened a replaced path");
    reader.ShutDown();
    reader.ShutDown();
    rejects([&] { reader.ReadBinaryView(1, 0); }, "Closed reader returned a view");
    require(reader.Initialize(next.c_str(), std::ios::binary | std::ios::in) &&
                std::string(reader.ReadBinaryView(4, 0), 4) == "next",
            "Reader could not be reinitialized");
    require(::truncate(next.c_str(), 0) == 0, "Cannot truncate test file");
    rejects([&] { reader.ReadBinaryView(4, 0); }, "Truncated mapping returned a view");
    rejects([&] { reader.ReadBinary(4, copy.data(), 0); }, "Truncated copying fallback was accepted");
    write_bytes(next, "next");
    write_bytes(root / "empty", "");
    std::filesystem::create_symlink(next, root / "symlink");
    require(::mkfifo((root / "fifo").c_str(), 0600) == 0, "Cannot create test FIFO");
    for (const char *name : {"empty", "symlink", "fifo", "absent"}) {
        require(!reader.Initialize((root / name).c_str(), std::ios::binary | std::ios::in),
                "Invalid posting file was accepted");
        rejects([&] { reader.ReadBinaryView(1, 0); }, "Failed initialization retained a stale map");
    }
    require(!reader.Initialize(root.c_str(), std::ios::binary | std::ios::in), "Directory was mapped");
    require(!reader.Initialize(next.c_str(), std::ios::binary | std::ios::out), "Write mode was accepted");
}

struct Counters {
    uint64_t views = 0;
    uint64_t copies = 0;
    uint64_t max_view_bytes = 0;
    int live = 0;
};

class CountingReader : public vortex_spfresh::PostingFileIO {
public:
    explicit CountingReader(std::shared_ptr<Counters> counters) : counters_(std::move(counters)) {
        ++counters_->live;
    }
    ~CountingReader() override {
        --counters_->live;
    }
    const char *ReadBinaryView(uint64_t size, uint64_t offset) override {
        ++counters_->views;
        counters_->max_view_bytes = std::max(counters_->max_view_bytes, size);
        return PostingFileIO::ReadBinaryView(size, offset);
    }
    uint64_t ReadBinary(uint64_t size, char *buffer, uint64_t offset = UINT64_MAX) override {
        ++counters_->copies;
        return PostingFileIO::ReadBinary(size, buffer, offset);
    }

private:
    std::shared_ptr<Counters> counters_;
};

struct ReaderFactory {
    decltype(SPTAG::SPANN::f_createAsyncIO) previous = SPTAG::SPANN::f_createAsyncIO;
    explicit ReaderFactory(const std::shared_ptr<Counters> &counters) {
        if (counters) {
            SPTAG::SPANN::f_createAsyncIO = [counters] {
                return std::make_shared<CountingReader>(counters);
            };
        } else {
            SPTAG::SPANN::f_createAsyncIO = [] {
                return std::make_shared<SPTAG::Helper::SimpleFileIO>();
            };
        }
    }
    ~ReaderFactory() {
        SPTAG::SPANN::f_createAsyncIO = std::move(previous);
    }
};

struct Close {
    void operator()(void *handle) const {
        vortex_spfresh_close(handle);
    }
};
using NativeHandle = std::unique_ptr<void, Close>;
using Hits = std::vector<std::pair<int, float>>;

struct Fixture {
    std::filesystem::path path;
    std::vector<float> vectors;
    static constexpr uint32_t dimension = 8;
    static constexpr uint32_t rows = 4096;
    static constexpr uint32_t pages = 12;
    static constexpr uint32_t heads = 32;

    explicit Fixture(const std::filesystem::path &root) : path(root), vectors(size_t(rows) * dimension) {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t component = 0; component < dimension; ++component) {
                vectors[size_t(row) * dimension + component] = float((row * 37 + component * 31) % 65521);
            }
        }
        char error[1024] = {};
        require(vortex_spfresh_build(path.c_str(), vectors.data(), dimension, rows, heads, pages, 4, error) ==
                    0,
                error);
    }

    NativeHandle open() const {
        void *handle = nullptr;
        char error[1024] = {};
        require(vortex_spfresh_open(path.c_str(), dimension, rows, pages, &handle, error) == 0, error);
        return NativeHandle(handle);
    }

    Hits search(void *handle, uint32_t row) const {
        std::array<int32_t, 10> ids {};
        std::array<float, 10> distances {};
        char error[1024] = {};
        require(vortex_spfresh_search(handle,
                                      vectors.data() + size_t(row) * dimension,
                                      dimension,
                                      1,
                                      ids.size(),
                                      8192,
                                      64,
                                      pages,
                                      ids.data(),
                                      distances.data(),
                                      error) == 0,
                error);
        Hits hits;
        for (size_t rank = 0; rank < ids.size(); ++rank) {
            hits.emplace_back(ids[rank], distances[rank]);
        }
        return hits;
    }
};

Hits search_postings(const Fixture &fixture,
                     const std::filesystem::path &root,
                     const std::shared_ptr<SPTAG::VectorIndex> &head,
                     const std::shared_ptr<Counters> &counters,
                     bool delta = false,
                     bool truth = false) {
    ReaderFactory factory(counters);
    SPTAG::SPANN::ExtraStaticSearcher<float> searcher;
    SPTAG::SPANN::Options options;
    options.m_indexDirectory = root;
    options.m_ssdIndex = "postings.bin";
    options.m_ioThreads = 1;
    options.m_searchPostingPageLimit = Fixture::pages;
    options.m_searchInternalResultNum = Fixture::heads;
    options.m_enableDataCompression = false;
    options.m_enableDeltaEncoding = delta;
    options.m_enablePostingListRearrange = false;
    SPTAG::COMMON::VersionLabel versions;
    require(searcher.LoadIndex(options, versions), "Cannot load posting test index");
    if (counters) {
        require(counters->live == 1, "Posting reader did not belong to the live searcher");
    }
    SPTAG::SPANN::ExtraWorkSpace workspace;
    workspace.Initialize(8192, 4, Fixture::heads, Fixture::pages * 4096, false);
    for (int posting = 0; posting < int(Fixture::heads); ++posting) {
        if (searcher.CheckValidPosting(posting)) {
            workspace.m_postingIDs.push_back(posting);
        }
    }
    require(!workspace.m_postingIDs.empty(), "Fixture has no populated postings");
    SPTAG::COMMON::QueryResultSet<float> query(fixture.vectors.data() + 7 * Fixture::dimension, 10);
    SPTAG::SPANN::SearchStats stats;
    std::set<int> targets {7, 8, 9};
    std::map<int, std::set<int>> found;
    searcher.SearchIndex(&workspace, query, head, &stats, truth ? &targets : nullptr, &found);
    require(stats.m_diskIOCount == workspace.m_postingIDs.size() && stats.m_diskAccessCount > 0,
            "Posting view changed logical read statistics");
    query.SortResult();
    Hits hits;
    for (int rank = 0; rank < query.GetResultNum(); ++rank) {
        const auto &hit = *query.GetResult(rank);
        hits.emplace_back(hit.VID, hit.Dist);
    }
    return hits;
}

void test_processing(const Fixture &fixture) {
    auto handle = fixture.open();
    auto head = vortex_spfresh_benchmark_index(handle.get()).GetMemoryIndex();
    const auto original_bytes = read_bytes(fixture.path / "postings.bin");
    auto counters = std::make_shared<Counters>();
    const auto expected = search_postings(fixture, fixture.path, head, nullptr);
    require(search_postings(fixture, fixture.path, head, counters) == expected,
            "Borrowed processing changed ranked IDs or distances");
    require(counters->views > 0 && counters->copies == 0 && counters->max_view_bytes > 4096,
            "Multi-page postings were not processed without copying");
    require(counters->live == 0, "Posting reader survived its owning searcher");
    for (const auto [delta, truth] : std::array<std::pair<bool, bool>, 2> {{{true, false}, {false, true}}}) {
        const auto control = search_postings(fixture, fixture.path, head, nullptr, delta, truth);
        counters = std::make_shared<Counters>();
        require(search_postings(fixture, fixture.path, head, counters, delta, truth) == control,
                "Writable/diagnostic copying fallback changed results");
        require(counters->views == 0 && counters->copies > 0 && counters->live == 0,
                "Writable/diagnostic processing borrowed a read-only view");
    }
    require(read_bytes(fixture.path / "postings.bin") == original_bytes,
            "Search modified immutable postings");
    const auto invalid = fixture.path.parent_path() / "invalid";
    std::filesystem::create_directory(invalid);
    for (bool misaligned : {false, true}) {
        write_bytes(invalid / "postings.bin", original_bytes);
        std::fstream file(invalid / "postings.bin", std::ios::binary | std::ios::in | std::ios::out);
        if (misaligned) {
            const uint16_t offset = 1;
            file.seekp(20);
            file.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
        } else {
            const int page = std::numeric_limits<int>::max() - 1;
            file.seekp(16);
            file.write(reinterpret_cast<const char *>(&page), sizeof(page));
        }
        file.close();
        rejects([&] { search_postings(fixture, invalid, head, std::make_shared<Counters>()); },
                "Malformed posting view was accepted");
    }
}

void test_bridge_toggle(const Fixture &fixture) {
    require(::setenv("VORTEX_SPFRESH_POSTING_VIEW", "0", 1) == 0, "Cannot disable posting views");
    auto copying = fixture.open();
    require(::setenv("VORTEX_SPFRESH_POSTING_VIEW", "1", 1) == 0, "Cannot enable posting views");
    auto mapped = fixture.open();
    for (uint32_t row : {7, 11, 31, 123, 1023, 2047, 4095}) {
        require(fixture.search(copying.get(), row) == fixture.search(mapped.get(), row),
                "Native handle A/B changed ranked IDs or distances");
    }
    require(::setenv("VORTEX_SPFRESH_POSTING_VIEW", "invalid", 1) == 0, "Cannot set invalid toggle");
    rejects([&] { fixture.open(); }, "Invalid posting-view toggle was accepted");
    require(fixture.search(copying.get(), 7) == fixture.search(mapped.get(), 7),
            "Changing the environment changed an already opened handle");
    require(::unsetenv("VORTEX_SPFRESH_POSTING_VIEW") == 0, "Cannot restore posting-view default");
    auto defaults = fixture.open();
    require(fixture.search(defaults.get(), 7) == fixture.search(mapped.get(), 7),
            "Default posting-view handle differs from explicit mode");
}
} // namespace

int main() {
    try {
        Scratch scratch;
        test_reader(scratch.path);
        Fixture fixture(scratch.path / "index");
        test_processing(fixture);
        test_bridge_toggle(fixture);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
