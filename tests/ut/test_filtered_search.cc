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

// get nb float vectors with sliced_dim dimensions, and return a dataset
inline knowhere::DataSetPtr
ReadDataset(const std::string& filename, int64_t nb = -1, int64_t sliced_dim = -1, std::string file_type = "default") {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    float* data = nullptr;

    if (file_type == "fvecs") {
        unsigned dim, num;
        in.read((char*)&dim, 4);             // 读取向量维度
        in.seekg(0, std::ios::end);          // 光标定位到文件末尾
        std::ios::pos_type ss = in.tellg();  // 获取文件大小（多少字节）
        size_t fsize = (size_t)ss;
        num = (unsigned)(fsize / (dim + 1) / 4);  // 数据的个数
        data = new float[(size_t)num * (size_t)dim];

        in.seekg(0, std::ios::beg);  // 光标定位到起始处
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);                 // 光标向右移动4个字节
            in.read((char*)(data + i * dim), dim * 4);  // 读取数据到一维数据data中
        }

        in.close();
        auto ret_ds = knowhere::GenDataSet(num, dim, data);
        ret_ds->SetIsOwner(true);
        return ret_ds;
    } else {
        int32_t real_dim;
        int32_t real_nb;
        in.read((char*)&real_nb, 4);
        in.read((char*)&real_dim, 4);

        int64_t dim = sliced_dim == -1 ? real_dim : sliced_dim;
        int64_t tmp_nb = nb == -1 ? real_nb : nb;

        data = new float[tmp_nb * dim];
        in.seekg(8, std::ios::beg);  // (num, dim)
        for (int32_t i = 0; i < tmp_nb; i++) {
            in.read((char*)(data + i * dim), dim * 4);
            if (dim < real_dim) {
                in.seekg((real_dim - dim) * 4, std::ios::cur);
            }
        }
        // for (size_t i = 0; i < nb * dim; i++) {
        //     std::cout << (float)data[i];
        //     if (!i) {
        //         std::cout << " ";
        //         continue;
        //     }
        //     if ((i + 1) % dim != 0) {
        //         std::cout << " ";
        //     } else {
        //         std::cout << std::endl;
        //     }
        // }
        in.close();
        auto ret_ds = knowhere::GenDataSet(tmp_nb, dim, data);
        ret_ds->SetIsOwner(true);
        return ret_ds;
    }
    return nullptr;
}

void
test_hnsw_filter() {
    int64_t nb = 200000, nq = 10000;
#ifdef DEBUG
    nb = 2000, nq = 10;
#endif
    const int64_t dim = 128;

    auto metric = std::string(knowhere::metric::L2);
    auto topk = 100;
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
    // TODO test with SIFT1M
    const std::string DATASET_FOLDER = "/home/ubuntu/kevin/dataset/sift/";
    const std::string base_file = DATASET_FOLDER + "sift_base.fvecs";
    const std::string query_file = DATASET_FOLDER + "sift_query.fvecs";
    const auto train_ds = ReadDataset(base_file, nb, dim, "fvecs");
    const auto query_ds = ReadDataset(query_file, nq, dim, "fvecs");
    // const auto train_ds = GenDataSet(nb, dim);
    // const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    // auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    using std::make_tuple;
    auto [name, gen, threshold] =
        make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFFilterThreshold);
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
    auto cfg_json = gen().dump();
    knowhere::Json json = knowhere::Json::parse(cfg_json);

    idx.Build(*train_ds, json);

    std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
        GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
    const auto filter_selectivity = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f};

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
            LOG_KNOWHERE_INFO_ << "=================";
        }
    }
}

int
main() {
    test_hnsw_filter();
    return 0;
}