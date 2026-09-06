#include <cstdint>
#include <list>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using TokenId = std::int64_t;

struct Merge {
    std::size_t rank;
    TokenId result;
};

struct PairHash {
    std::size_t operator()(const std::pair<TokenId, TokenId>& pair) const noexcept {
        const auto left = std::hash<TokenId>{}(pair.first);
        const auto right = std::hash<TokenId>{}(pair.second);
        return left ^ (right + 0x9e3779b97f4a7c15ULL + (left << 6U) + (left >> 2U));
    }
};

class BPEEngine {
  public:
    BPEEngine(const py::dict& vocab, const py::list& merges) {
        std::unordered_map<std::string, TokenId> token_to_id;
        token_to_id.reserve(vocab.size());

        for (const auto& item : vocab) {
            const auto token_id = py::cast<TokenId>(item.first);
            const auto token = py::cast<std::string>(item.second);
            token_to_id.emplace(token, token_id);
            if (token.size() == 1) {
                byte_to_id_[static_cast<unsigned char>(token.front())] = token_id;
                has_byte_[static_cast<unsigned char>(token.front())] = true;
            }
        }

        merge_table_.reserve(merges.size());
        std::size_t rank = 0;
        for (const auto& item : merges) {
            const auto pair = py::cast<py::tuple>(item);
            const auto left = py::cast<std::string>(pair[0]);
            const auto right = py::cast<std::string>(pair[1]);

            const auto left_it = token_to_id.find(left);
            const auto right_it = token_to_id.find(right);
            const auto result_it = token_to_id.find(left + right);
            if (left_it == token_to_id.end() || right_it == token_to_id.end() || result_it == token_to_id.end()) {
                throw std::invalid_argument("A merge references a token missing from the vocabulary");
            }

            merge_table_.emplace(
                std::make_pair(left_it->second, right_it->second),
                Merge{rank, result_it->second}
            );
            ++rank;
        }
    }

    std::vector<TokenId> encode_pretokens(const std::vector<std::string>& pretokens) const {
        std::vector<TokenId> output;
        std::size_t byte_count = 0;
        for (const auto& pretoken : pretokens) {
            byte_count += pretoken.size();
        }
        output.reserve(byte_count);

        for (const auto& pretoken : pretokens) {
            encode_one(pretoken, output);
        }
        return output;
    }

  private:
    void encode_one(const std::string& pretoken, std::vector<TokenId>& output) const {
        std::vector<TokenId> pieces;
        pieces.reserve(pretoken.size());
        for (const unsigned char byte : pretoken) {
            if (!has_byte_[byte]) {
                throw std::runtime_error("The vocabulary does not contain every byte token");
            }
            pieces.push_back(byte_to_id_[byte]);
        }

        while (pieces.size() > 1) {
            std::size_t best_index = 0;
            std::size_t best_rank = std::numeric_limits<std::size_t>::max();
            TokenId merged_id = 0;

            for (std::size_t index = 0; index + 1 < pieces.size(); ++index) {
                const auto merge_it = merge_table_.find({pieces[index], pieces[index + 1]});
                if (merge_it != merge_table_.end() && merge_it->second.rank < best_rank) {
                    best_rank = merge_it->second.rank;
                    best_index = index;
                    merged_id = merge_it->second.result;
                }
            }

            if (best_rank == std::numeric_limits<std::size_t>::max()) {
                break;
            }

            pieces[best_index] = merged_id;
            pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
        }

        output.insert(output.end(), pieces.begin(), pieces.end());
    }

    TokenId byte_to_id_[256]{};
    bool has_byte_[256]{};
    std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merge_table_;
};

struct CachedNode {
    TokenId value;
    std::int32_t prev;
    std::int32_t next;
    std::uint32_t generation = 0;
    bool alive = true;
};

struct CachedCandidate {
    std::size_t rank;
    std::size_t position;
    std::int32_t left;
    std::int32_t right;
    std::uint32_t left_generation;
    std::uint32_t right_generation;
};

struct CachedCandidateGreater {
    bool operator()(const CachedCandidate& lhs, const CachedCandidate& rhs) const noexcept {
        if (lhs.rank != rhs.rank) {
            return lhs.rank > rhs.rank;
        }
        return lhs.position > rhs.position;
    }
};

class CachedBPEEngine {
  public:
    CachedBPEEngine(const py::dict& vocab, const py::list& merges, std::size_t cache_capacity = 65'536)
        : cache_capacity_(cache_capacity) {
        std::unordered_map<std::string, TokenId> token_to_id;
        token_to_id.reserve(vocab.size());

        for (const auto& item : vocab) {
            const auto token_id = py::cast<TokenId>(item.first);
            const auto token = py::cast<std::string>(item.second);
            token_to_id.emplace(token, token_id);
            if (token.size() == 1) {
                const auto byte = static_cast<unsigned char>(token.front());
                byte_to_id_[byte] = token_id;
                has_byte_[byte] = true;
            }
        }

        merge_table_.reserve(merges.size());
        std::size_t rank = 0;
        for (const auto& item : merges) {
            const auto pair = py::cast<py::tuple>(item);
            const auto left = py::cast<std::string>(pair[0]);
            const auto right = py::cast<std::string>(pair[1]);

            const auto left_it = token_to_id.find(left);
            const auto right_it = token_to_id.find(right);
            const auto result_it = token_to_id.find(left + right);
            if (left_it == token_to_id.end() || right_it == token_to_id.end() || result_it == token_to_id.end()) {
                throw std::invalid_argument("A merge references a token missing from the vocabulary");
            }

            merge_table_.emplace(
                std::make_pair(left_it->second, right_it->second),
                Merge{rank, result_it->second}
            );
            ++rank;
        }
    }

    std::vector<TokenId> encode_pretokens(const std::vector<std::string>& pretokens) {
        std::vector<TokenId> output;
        std::size_t byte_count = 0;
        for (const auto& pretoken : pretokens) {
            byte_count += pretoken.size();
        }
        output.reserve(byte_count);

        for (const auto& pretoken : pretokens) {
            if (append_cached(pretoken, output)) {
                continue;
            }

            std::vector<TokenId> encoded;
            encoded.reserve(pretoken.size());
            encode_one(pretoken, encoded);
            output.insert(output.end(), encoded.begin(), encoded.end());
            store_cached(pretoken, encoded);
        }
        return output;
    }

    std::pair<std::size_t, std::size_t> cache_stats() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return {cache_hits_, cache_misses_};
    }

    std::size_t cache_size() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_.size();
    }

    void clear_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
        cache_order_.clear();
        cache_hits_ = 0;
        cache_misses_ = 0;
    }

  private:
    using CandidateHeap = std::priority_queue<CachedCandidate, std::vector<CachedCandidate>, CachedCandidateGreater>;

    struct CacheValue {
        std::vector<TokenId> token_ids;
        std::list<std::string>::iterator order_iterator;
    };

    const Merge* find_merge(TokenId left, TokenId right) const {
        const auto merge_it = merge_table_.find({left, right});
        return merge_it == merge_table_.end() ? nullptr : &merge_it->second;
    }

    void push_candidate(CandidateHeap& heap, const std::vector<CachedNode>& nodes, std::int32_t left_index) const {
        if (left_index < 0) {
            return;
        }

        const auto& left = nodes[static_cast<std::size_t>(left_index)];
        if (!left.alive || left.next < 0) {
            return;
        }

        const auto right_index = left.next;
        const auto& right = nodes[static_cast<std::size_t>(right_index)];
        const auto* merge = find_merge(left.value, right.value);
        if (merge == nullptr) {
            return;
        }

        heap.push(CachedCandidate{
            merge->rank,
            static_cast<std::size_t>(left_index),
            left_index,
            right_index,
            left.generation,
            right.generation,
        });
    }

    bool candidate_is_valid(const CachedCandidate& candidate, const std::vector<CachedNode>& nodes) const {
        const auto& left = nodes[static_cast<std::size_t>(candidate.left)];
        const auto& right = nodes[static_cast<std::size_t>(candidate.right)];
        if (!left.alive || !right.alive || left.next != candidate.right || right.prev != candidate.left) {
            return false;
        }
        if (left.generation != candidate.left_generation || right.generation != candidate.right_generation) {
            return false;
        }
        const auto* merge = find_merge(left.value, right.value);
        return merge != nullptr && merge->rank == candidate.rank;
    }

    void encode_one(const std::string& pretoken, std::vector<TokenId>& output) const {
        if (pretoken.empty()) {
            return;
        }

        std::vector<CachedNode> nodes;
        nodes.reserve(pretoken.size());
        for (std::size_t index = 0; index < pretoken.size(); ++index) {
            const auto byte = static_cast<unsigned char>(pretoken[index]);
            if (!has_byte_[byte]) {
                throw std::runtime_error("The vocabulary does not contain every byte token");
            }
            nodes.push_back(CachedNode{
                byte_to_id_[byte],
                index == 0 ? -1 : static_cast<std::int32_t>(index - 1),
                index + 1 == pretoken.size() ? -1 : static_cast<std::int32_t>(index + 1),
            });
        }

        CandidateHeap heap;
        for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
            push_candidate(heap, nodes, static_cast<std::int32_t>(index));
        }

        while (!heap.empty()) {
            const auto candidate = heap.top();
            heap.pop();
            if (!candidate_is_valid(candidate, nodes)) {
                continue;
            }

            auto& left = nodes[static_cast<std::size_t>(candidate.left)];
            auto& right = nodes[static_cast<std::size_t>(candidate.right)];
            const auto* merge = find_merge(left.value, right.value);

            left.value = merge->result;
            ++left.generation;
            left.next = right.next;
            right.alive = false;
            ++right.generation;

            if (left.next >= 0) {
                auto& next = nodes[static_cast<std::size_t>(left.next)];
                next.prev = candidate.left;
            }

            push_candidate(heap, nodes, left.prev);
            push_candidate(heap, nodes, candidate.left);
        }

        std::int32_t index = 0;
        while (index >= 0) {
            const auto& node = nodes[static_cast<std::size_t>(index)];
            output.push_back(node.value);
            index = node.next;
        }
    }

    bool append_cached(const std::string& pretoken, std::vector<TokenId>& output) {
        if (cache_capacity_ == 0) {
            return false;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);
        const auto cache_it = cache_.find(pretoken);
        if (cache_it == cache_.end()) {
            ++cache_misses_;
            return false;
        }

        cache_order_.splice(cache_order_.begin(), cache_order_, cache_it->second.order_iterator);
        output.insert(output.end(), cache_it->second.token_ids.begin(), cache_it->second.token_ids.end());
        ++cache_hits_;
        return true;
    }

    void store_cached(const std::string& pretoken, const std::vector<TokenId>& token_ids) {
        if (cache_capacity_ == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);
        const auto existing = cache_.find(pretoken);
        if (existing != cache_.end()) {
            existing->second.token_ids = token_ids;
            cache_order_.splice(cache_order_.begin(), cache_order_, existing->second.order_iterator);
            return;
        }

        cache_order_.push_front(pretoken);
        cache_.emplace(pretoken, CacheValue{token_ids, cache_order_.begin()});
        if (cache_.size() > cache_capacity_) {
            const auto& oldest = cache_order_.back();
            cache_.erase(oldest);
            cache_order_.pop_back();
        }
    }

    TokenId byte_to_id_[256]{};
    bool has_byte_[256]{};
    std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merge_table_;
    std::size_t cache_capacity_;
    mutable std::mutex cache_mutex_;
    std::list<std::string> cache_order_;
    std::unordered_map<std::string, CacheValue> cache_;
    std::size_t cache_hits_ = 0;
    std::size_t cache_misses_ = 0;
};

}  // namespace

PYBIND11_MODULE(_bpe_native, module) {
    module.doc() = "Native BPE merge engine for CS336";
    py::class_<BPEEngine>(module, "BPEEngine")
        .def(py::init<const py::dict&, const py::list&>())
        .def(
            "encode_pretokens",
            &BPEEngine::encode_pretokens,
            py::call_guard<py::gil_scoped_release>()
        );
    py::class_<CachedBPEEngine>(module, "CachedBPEEngine")
        .def(
            py::init<const py::dict&, const py::list&, std::size_t>(),
            py::arg("vocab"),
            py::arg("merges"),
            py::arg("cache_capacity") = 65'536
        )
        .def(
            "encode_pretokens",
            &CachedBPEEngine::encode_pretokens,
            py::call_guard<py::gil_scoped_release>()
        )
        .def("cache_stats", &CachedBPEEngine::cache_stats)
        .def("cache_size", &CachedBPEEngine::cache_size)
        .def("clear_cache", &CachedBPEEngine::clear_cache);
}
