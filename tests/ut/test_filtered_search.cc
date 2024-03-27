#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.95f;
}  // namespace

void
test_hnsw_filter() {
    int64_t nb = 200000, nq = 1000;
#ifdef DEBUG
    nb = 2000, nq = 10;
#endif
    const int64_t dim = 128;

    auto metric = std::string(knowhere::metric::L2);
    auto topk = 10;
    auto version = static_cast<int32_t>(knowhere::Version::GetCurrentVersion().VersionNumber());

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto hnsw_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 32;
        json[knowhere::indexparam::EFCONSTRUCTION] = 120;
        json[knowhere::indexparam::EF] = 120;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    using std::make_tuple;
    auto [name, gen, threshold] =
        make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFFilterThreshold);
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
    auto cfg_json = gen().dump();
    knowhere::Json json = knowhere::Json::parse(cfg_json);

    idx.Build(*train_ds, json);

    std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
        GenerateBitsetWithFirstTbitsSet};
    const auto filter_selectivity = {0.1f, 0.5f, 0.9f};

    for (const float selectivity : filter_selectivity) {
        for (const auto& gen_func : gen_bitset_funcs) {
            auto bitset_data = gen_func(nb, nb * selectivity);
            knowhere::BitsetView bitset(bitset_data.data(), nb);

            knowhere::TimeRecorder tr(std::string("HNSW filter search"), 2);
            auto results = idx.Search(*query_ds, json, bitset);
            double latency = tr.ElapseFromBegin("HNSW filter search");
            latency *= 0.000001;  // convert to s
            double qps = nq / latency;

            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, bitset);
            float recall = GetKNNRecall(*gt.value(), *results.value());

            // assert(recall > kKnnRecallThreshold);
            LOG_KNOWHERE_INFO_ << "selectivity: " << selectivity << ", recall: " << recall << ", qps: " << qps;
        }
    }
}

int
main() {
    test_hnsw_filter();
    return 0;
}