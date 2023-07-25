// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "utils.h"

TEST_CASE("Test Binary Get Vector By Ids", "[Binary GetVectorByIds]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 100;
    const int64_t dim = 128;

    const auto metric_type = knowhere::metric::HAMMING;

    auto base_bin_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric_type;
        json[knowhere::meta::TOPK] = 1;
        return json;
    };

    auto bin_ivfflat_gen = [&base_bin_gen]() {
        knowhere::Json json = base_bin_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 4;
        return json;
    };

    auto bin_hnsw_gen = [&base_bin_gen]() {
        knowhere::Json json = base_bin_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 64;
        return json;
    };

    auto bin_flat_gen = base_bin_gen;

    SECTION("Test binary index") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, bin_flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, bin_ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, bin_hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        if (!idx.HasRawData(metric_type)) {
            return;
        }
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenBinDataSet(nb, dim);
        auto ids_ds = GenIdsDataSet(nb, nq);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_new = knowhere::IndexFactory::Instance().Create(name);
        idx_new.Deserialize(bs);
        auto results = idx_new.GetVectorByIds(*ids_ds);
        REQUIRE(results.has_value());
        auto xb = (uint8_t*)train_ds->GetTensor();
        auto res_rows = results.value()->GetRows();
        auto res_dim = results.value()->GetDim();
        auto res_data = (uint8_t*)results.value()->GetTensor();
        REQUIRE(res_rows == nq);
        REQUIRE(res_dim == dim);
        const auto data_bytes = dim / 8;
        for (int i = 0; i < nq; ++i) {
            auto id = ids_ds->GetIds()[i];
            for (int j = 0; j < data_bytes; ++j) {
                REQUIRE(res_data[i * data_bytes + j] == xb[id * data_bytes + j]);
            }
        }
    }
}

TEST_CASE("Test Float Get Vector By Ids", "[Float GetVectorByIds]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 100;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = 1;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 32;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 4;
        return json;
    };

    auto ivfflatcc_gen = [&ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::SSIZE] = 48;
        return json;
    };

    auto flat_gen = base_gen;

    auto load_raw_data = [](knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSet& dataset,
                            const knowhere::Json& conf) {
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto p_data = dataset.GetTensor();
        knowhere::BinarySet bs;
        auto res = index.Serialize(bs);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
        bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
        bptr->size = dim * rows * sizeof(float);
        bs.Append("RAW_DATA", bptr);
        res = index.Deserialize(bs);
        REQUIRE(res == knowhere::Status::success);
    };

    SECTION("Test float index") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, base_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, base_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        if (!idx.HasRawData(metric)) {
            return;
        }
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim);
        auto train_ds_copy = CopyDataSet(train_ds, nb);
        auto ids_ds = GenIdsDataSet(nb, nq);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_new = knowhere::IndexFactory::Instance().Create(name);
        idx_new.Deserialize(bs);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx_new, *train_ds, json);
        }
        auto results = idx_new.GetVectorByIds(*ids_ds);
        REQUIRE(results.has_value());
        auto xb = (float*)train_ds_copy->GetTensor();
        auto res_rows = results.value()->GetRows();
        auto res_dim = results.value()->GetDim();
        auto res_data = (float*)results.value()->GetTensor();
        REQUIRE(res_rows == nq);
        REQUIRE(res_dim == dim);
        for (int i = 0; i < nq; ++i) {
            const auto id = ids_ds->GetIds()[i];
            for (int j = 0; j < dim; ++j) {
                REQUIRE(res_data[i * dim + j] == xb[id * dim + j]);
            }
        }
    }
}
