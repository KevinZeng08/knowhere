#include <iostream>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/kmeans.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
#include "utils.h"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))

inline knowhere::DataSetPtr
ReadDataset(const std::string& file_path, int64_t nb = -1) {
    std::ifstream in(file_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    int32_t real_dim;
    int32_t real_nb;
    in.read((char*)&real_nb, 4);
    in.read((char*)&real_dim, 4);

    if (nb != -1) {
        real_nb = nb;
    }
    LOG_KNOWHERE_INFO_ << "# of points: " << real_nb << ", dim: " << real_dim;
    float* data = new float[(int64_t)real_nb * (int64_t)real_dim];
    in.seekg(8, std::ios::beg);  // (num, dim)
    for (int64_t i = 0; i < real_nb; ++i) {
        in.read((char*)(data + i * real_dim), real_dim * sizeof(float));
    }
    in.close();
    auto ret_ds = knowhere::GenDataSet(real_nb, real_dim, data);
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline void
ReadGt(const std::string& file_path, int64_t* gt_ids) {
    std::ifstream in(file_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    int32_t nq;
    int32_t topk;
    in.read((char*)&nq, 4);
    in.read((char*)&topk, 4);
    int32_t* ids = new int32_t[nq * topk];
    for (int i = 0; i < nq; ++i) {
        in.read((char*)(ids + i * topk), topk * sizeof(int32_t));
    }
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < topk; ++j) {
            gt_ids[i * topk + j] = ids[i * topk + j];
        }
    }
    in.close();
}

inline float
L2Sqr(const float* x, const float* y, size_t dim) {
    float dist = 0.0;
    for (size_t i = 0; i < dim; i++) {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return dist;
}

static float
L2SqrSIMD16ExtSSE(const float* pVect1v, const float* pVect2v, const size_t qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = qty_ptr;
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
struct CompareByFirst {
    bool
    operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    }
};

const std::string base_file = "/home/ubuntu/gao/openai.fbin";
const std::string query_file = "/home/ubuntu/gao/openai_query.fbin";
const std::string gt_file = "/home/ubuntu/gao/openai_gt.fbin";
const size_t num_clusters_level1 = 128;
const size_t num_clusters_level2 = 1024;
const size_t min_points_per_centroid = 39;
#define KMEANS_LEVEL_1
// #define KMEANS_LEVEL_2
// #define TEST

#define READ_DATA
// read data
size_t topk = 100;
#ifdef TEST
    auto train_ds = GenDataSet(1000000, 128);
    auto query_ds = GenDataSet(10, 128);
    // groundtruth
    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
    const int64_t* gt_ids = (gt.value())->GetIds();
#else
    auto train_ds = ReadDataset(base_file, 4000000);
    auto query_ds = ReadDataset(query_file);
    // groundtruth
    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
    const int64_t* gt_ids = (gt.value())->GetIds();
    // int64_t* gt_ids = new int64_t[query_ds->GetRows() * topk];
    // ReadGt(gt_file, gt_ids);
#endif

READ_DATA

void
two_level_kmeans() {

    size_t K1 = num_clusters_level1;
    float* base_vecs = (float*)train_ds->GetTensor();
    int64_t nb = train_ds->GetRows();
    int64_t dim = train_ds->GetDim();
    float* query_vecs = (float*)query_ds->GetTensor();
    int64_t nq = query_ds->GetRows();
    auto kmeans = knowhere::kmeans::KMeans<float>(K1, dim);
    kmeans.fit(base_vecs, nb);
    kmeans.calculate_result_ids(nb);

    // 1 level kmeans
    // query 1 level, calculate recall and search data size
    auto& centroids = kmeans.get_centroids();
    auto& result_ids = kmeans.get_result_ids();
#ifdef KMEANS_LEVEL_1
    std::ofstream out("1level_result_" + std::to_string(num_clusters_level1) + ".txt");
    out << "search_ratio,recall,search_data_size" << std::endl;
    std::vector<float> search_ratios{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
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
                float dist = L2SqrSIMD16ExtSSE(query_vecs + i * dim, centroids.get() + j * dim, dim);
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
        out << search_ratio << "," << recall << "," << ratio << std::endl;
    }
#endif
    // 2-level kmeans

    // for each 1 level, 2 level kmeans
#ifdef KMEANS_LEVEL_2
    // store the mapping from each 2 level sub-cluster to corresponding 1level id
    std::vector<std::vector<uint32_t>> id_mapping;
    id_mapping.resize(K1);
    // #pragma omp parallel for schedule(dynamic) Kmeans.fit() is already parallel
    knowhere::TimeRecorder tr("2-level kmeans", 2);
    std::vector<knowhere::kmeans::KMeans<float>> kmeans_pool;
    for (size_t i = 0; i < K1; ++i) {
        size_t n = result_ids[i].size();
        id_mapping[i].resize(n);
        int num = std::min(num_clusters_level2, n / min_points_per_centroid);
        kmeans_pool.emplace_back(num, dim);
        auto& kmeans = kmeans_pool[i];
        auto vecs = std::make_unique<float[]>(n * dim);
        for (size_t j = 0; j < n; ++j) {
            id_mapping[i][j] = result_ids[i][j];
            memcpy(vecs.get() + j * dim, base_vecs + result_ids[i][j] * dim, dim * sizeof(float));
        }
        kmeans.fit(vecs.get(), n);
        kmeans.calculate_result_ids(n);
    }
    tr.ElapseFromBegin("finish 2-level kmeans");
    // query 2 level, calculate recall
    std::ofstream out2("2level_result_" + std::to_string(num_clusters_level1) + ".txt");
    out2 << "search_ratio1,search_ratio2,recall,search_data_size" << std::endl;
    std::vector<float> search_ratios_level1{0.01, 0.02, 0.05, 0.1};
    std::vector<float> search_ratios_level2{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    for (float search_ratio1 : search_ratios_level1) {
        for (float search_ratio2 : search_ratios_level2) {
            size_t nprobe1 = search_ratio1 * K1;
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
                    float dist = L2SqrSIMD16ExtSSE(query_vecs + i * dim, centroids.get() + j * dim, dim);
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
                for (size_t j = 0; j < nprobe1; j++) {
                    auto& kmeans = kmeans_pool[top_centroid_ids[j]];
                    size_t K2 = kmeans.get_n_centroids();
                    size_t nprobe2 = search_ratio2 * K2;
                    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
                        top_centroids2;
                    for (size_t k = 0; k < K2; ++k) {
                        float dist =
                            L2SqrSIMD16ExtSSE(query_vecs + i * dim, kmeans.get_centroids().get() + k * dim, dim);
                        top_centroids2.push(std::make_pair(dist, k));
                        if (top_centroids2.size() > nprobe2) {
                            top_centroids2.pop();
                        }
                    }
                    std::vector<int> top_centroid_ids2(nprobe2);
                    for (size_t k = 0; k < nprobe2; ++k) {
                        top_centroid_ids2[nprobe2 - k - 1] = top_centroids2.top().second;
                        top_centroids2.pop();
                    }
                    // search buckets
                    size_t n = result_ids[top_centroid_ids[j]].size();
                    auto& result_ids = kmeans_pool[top_centroid_ids[j]].get_result_ids();
                    for (size_t l = 0; l < nprobe2; ++l) {
                        auto& ids = result_ids[top_centroid_ids2[l]];
                        points[i] += ids.size();
                        for (auto id : ids) {
                            if (gt_id_set.find(id_mapping[top_centroid_ids[j]][id]) != gt_id_set.end()) {
                                corrects[i]++;
                            }
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
            out2 << search_ratio1 << "," << search_ratio2 << "," << recall << "," << ratio << std::endl;
        }
    }
#endif
}

void
one_level_kmeans() {

    float* base_vecs = (float*)train_ds->GetTensor();
    int64_t nb = train_ds->GetRows();
    int64_t dim = train_ds->GetDim();
    float* query_vecs = (float*)query_ds->GetTensor();
    int64_t nq = query_ds->GetRows();
    // equally partition the dataset into num_clusters_level1 partitions
    size_t K1 = num_clusters_level1;

    std::vector<std::vector<uint32_t>> id_mapping;

    knowhere::TimeRecorder tr("1-level kmeans", 2);
    std::vector<knowhere::kmeans::KMeans<float>> kmeans_pool;
    // partition dataset into K1 parts
    for (size_t i = 0; i < nb; i += nb / K1) {
        size_t start = i;
        size_t end = std::min(i + nb / K1, (size_t)nb);
        size_t n = end - start;
        int num = std::min(num_clusters_level2, n / min_points_per_centroid);
        if (num == 0) {
            num = 1;
        }
        kmeans_pool.emplace_back(num, dim);
        id_mapping.emplace_back();
        auto& kmeans = kmeans_pool.back();
        auto vecs = std::make_unique<float[]>(n * dim);
        for (size_t j = 0; j < n; ++j) {
            id_mapping.back().push_back(start + j);
            memcpy(vecs.get() + j * dim, base_vecs + (start + j) * dim, dim * sizeof(float));
        }
        kmeans.fit(vecs.get(), n);
        kmeans.calculate_result_ids(n);
    }
    tr.ElapseFromBegin("finish 1-level kmeans");
    // query 1 level, calculate recall
    std::ofstream out("without_1level_result_" + std::to_string(num_clusters_level1) + ".txt");
    out << "seach_ratio1,search_ratio2,recall,search_data_size" << std::endl;
    std::vector<float> search_ratios_level1{1};
    std::vector<float> search_ratios_level2{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
    for (float search_ratio1 : search_ratios_level1) {
        for (float search_ratio2 : search_ratios_level2) {
            size_t nprobe1 = search_ratio1 * K1;
            std::vector<int> corrects(query_ds->GetRows(), 0);
            std::vector<int64_t> points(query_ds->GetRows(), 0);

            omp_set_num_threads(8);
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < nq; ++i) {
                const int64_t* gt_id = gt_ids + i * topk;
                std::unordered_set<int64_t> gt_id_set;
                for (size_t j = 0; j < topk; ++j) {
                    gt_id_set.insert(gt_id[j]);
                }
                // level2
                for (size_t j = 0; j < nprobe1; j++) {
                    auto& kmeans = kmeans_pool[j];
                    size_t K2 = kmeans.get_n_centroids();
                    size_t nprobe2 = search_ratio2 * K2;
                    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
                        top_centroids2;
                    for (size_t k = 0; k < K2; ++k) {
                        float dist =
                            L2SqrSIMD16ExtSSE(query_vecs + i * dim, kmeans.get_centroids().get() + k * dim, dim);
                        top_centroids2.push(std::make_pair(dist, k));
                        if (top_centroids2.size() > nprobe2) {
                            top_centroids2.pop();
                        }
                    }
                    std::vector<int> top_centroid_ids2(nprobe2);
                    for (size_t k = 0; k < nprobe2; ++k) {
                        top_centroid_ids2[nprobe2 - k - 1] = top_centroids2.top().second;
                        top_centroids2.pop();
                    }
                    // search buckets
                    auto& result_ids = kmeans_pool[j].get_result_ids();
                    for (size_t l = 0; l < nprobe2; ++l) {
                        auto& ids = result_ids[top_centroid_ids2[l]];
                        points[i] += ids.size();
                        for (auto id : ids) {
                            if (gt_id_set.find(id_mapping[j][id]) != gt_id_set.end()) {
                                corrects[i]++;
                            }
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
            out << search_ratio1 << "," << search_ratio2 << "," << recall << "," << ratio << std::endl;
        }
    }
}

int
main(int argc, char** argv) {
    two_level_kmeans();
    // one_level_kmeans();
    return 0;
}
