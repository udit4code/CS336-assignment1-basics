#include <cstdint>
#include <limits>
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
}
