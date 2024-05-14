#include <iostream>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/kmeans.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "utils.h"

inline knowhere::DataSetPtr
ReadDataset(const std::string& file_path, int64_t& nb) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
}

inline float
L2Sqr(const float* x, const float* y, size_t dim) {
    float dist = 0.0;
    for (size_t i = 0; i < dim; i++) {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return dist;
}

struct CompareByFirst {
    bool
    operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    }
};

const std::string base_file = "";
const std::string query_file = "";
// #define KMEANS_LEVEL_1
#define KMEANS_LEVEL_2

int
main(int argc, char** argv) {
    // read data
    size_t topk = 100;
    // auto train_ds = ReadDataset(base_file);
    auto train_ds = GenDataSet(100000, 128);
    // auto query_ds = ReadDataset(query_file);
    auto query_ds = GenDataSet(100, 128);

    // groundtruth
    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
    const int64_t* gt_ids = (gt.value())->GetIds();

    size_t K1 = 100;
    float* base_vecs = (float*)train_ds->GetTensor();
    int64_t nb = train_ds->GetRows();
    int64_t dim = train_ds->GetDim();
    float* query_vecs = (float*)query_ds->GetTensor();
    int64_t nq = query_ds->GetRows();
    auto kmeans = knowhere::kmeans::KMeans<float>(K1, dim);
    kmeans.fit(base_vecs, nb);

    // 1 level kmeans
    // query 1 level, calculate recall and search data size
    auto& centroids = kmeans.get_centroids();
    auto& result_ids = kmeans.get_result_ids(nb);
#ifdef KMEANS_LEVEL_1
    std::ofstream out("1level_result.txt");
    out << "recall,search_data_size" << std::endl;
    std::vector<float> search_ratios{0.2, 0.4, 0.6, 0.8, 1.0};
    for (float search_ratio : search_ratios) {
        size_t nprobe1 = search_ratio * K1;
        std::vector<int> corrects(nq, 0);
        std::vector<int64_t> points(nq, 0);

        omp_set_num_threads(8);
        // find closest nprobe centroids, search in these buckets
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nq; ++i) {
            const int64_t* gt_id = gt_ids + i * topk;
            std::unordered_set<int64_t> gt_id_set;
            for (size_t j = 0; j < topk; ++j) {
                gt_id_set.insert(gt_id[j]);
            }
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
                top_centroids;
            for (size_t j = 0; j < K1; ++j) {
                float dist = L2Sqr(query_vecs + i * dim, centroids.get() + j * dim, dim);
                top_centroids.push(std::make_pair(dist, j));
                if (top_centroids.size() > nprobe1) {
                    top_centroids.pop();
                }
            }
            std::vector<int> top_centroid_ids(nprobe1);
            for (size_t j = 0; j < nprobe1; ++j) {
                top_centroid_ids[nprobe1 - j - 1] = top_centroids.top().second;
                top_centroids.pop();
            }
            // search buckets
            for (size_t j = 0; j < nprobe1; ++j) {
                auto& ids = result_ids[top_centroid_ids[j]];
                points[i] += ids.size();
                for (auto id : ids) {
                    if (gt_id_set.find(id) != gt_id_set.end()) {
                        corrects[i]++;
                    }
                }
            }
        }
        // recall - search data size
        float recall = 0.0;
        float ratio = 0.0;
        int total_correct = 0;
        int total_points = 0;
        for (auto correct : corrects) {
            total_correct += correct;
        }
        recall = (float)total_correct / (nq * topk);
        for (auto point : points) {
            total_points += point;
        }
        ratio = (float)total_points / nq / nb;
        out << recall << "," << ratio << std::endl;
    }
    #endif
    // 2-level kmeans
    
    // for each 1 level, 2 level kmeans
#ifdef KMEANS_LEVEL_2
    size_t K2 = 10;
    std::vector<knowhere::kmeans::KMeans<float>> kmeans_pool;
    for (size_t i = 0; i < K1; ++i) {
        kmeans_pool.emplace_back(K2, dim);
    }
    // store the mapping from each 2 level sub-cluster to corresponding 1level id
    std::vector<std::vector<uint32_t>> id_mapping;
    id_mapping.resize(K1);
// #pragma omp parallel for schedule(dynamic) Kmeans.fit() is already parallel
    for (size_t i = 0; i < K1; ++i) {
        auto& kmeans = kmeans_pool[i];
        size_t n = result_ids[i].size();
        id_mapping[i].resize(n);
        auto vecs = std::make_unique<float[]>(n * dim);
        for (size_t j = 0; j < n; ++j) {
            id_mapping[i][j] = result_ids[i][j];
            memcpy(vecs.get() + j * dim, base_vecs + result_ids[i][j] * dim, dim * sizeof(float));
        }
        kmeans.fit(vecs.get(), n);
    }
    // query 2 level, calculate recall
    std::ofstream out2("2level_result.txt");
    out2 << "recall,search_data_size" << std::endl;
    std::vector<float> search_ratios_level1 {0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<float> search_ratios_level2 {0.2, 0.4, 0.6, 0.8, 1.0};
    
    for (float search_ratio1 : search_ratios_level1) {
        for (float search_ratio2 : search_ratios_level2) {
            size_t nprobe1 = search_ratio1 * K1;
            size_t nprobe2 = search_ratio2 * K2;
            std::vector<int> corrects(nq, 0);
            std::vector<int64_t> points(nq, 0);

            omp_set_num_threads(8);
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < nq; ++i) {
                const int64_t* gt_id = gt_ids + i * topk;
                std::unordered_set<int64_t> gt_id_set;
                for (size_t j = 0; j < topk; ++j) {
                    gt_id_set.insert(gt_id[j]);
                }
                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
                    top_centroids;
                for (size_t j = 0; j < K1; ++j) {
                    float dist = L2Sqr(query_vecs + i * dim, centroids.get() + j * dim, dim);
                    top_centroids.push(std::make_pair(dist, j));
                    if (top_centroids.size() > nprobe1) {
                        top_centroids.pop();
                    }
                }
                std::vector<int> top_centroid_ids(nprobe1);
                for (size_t j = 0; j < nprobe1; ++j) {
                    top_centroid_ids[nprobe1 - j - 1] = top_centroids.top().second;
                    top_centroids.pop();
                }
                // level2
                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
                    top_centroids2;
                for (size_t j = 0; j < K2; ++j) {
                    float dist = L2Sqr(query_vecs + i * dim, kmeans_pool[top_centroid_ids[0]].get_centroids().get() + j * dim, dim);
                    top_centroids2.push(std::make_pair(dist, j));
                    if (top_centroids2.size() > nprobe2) {
                        top_centroids2.pop();
                    }
                }
                std::vector<int> top_centroid_ids2(nprobe2);
                for (size_t j = 0; j < nprobe2; ++j) {
                    top_centroid_ids2[nprobe2 - j - 1] = top_centroids2.top().second;
                    top_centroids2.pop();
                }
                // TODO search buckets
                for (size_t j = 0; j < nprobe1; ++j) {
                    
                }
            }
        }
    }
#endif
    return 0;
}
