#include "knowhere/kmeans.h"
#include "knowhere/utils.h"
#include "utils.h"

#include <iostream>

inline knowhere::DataSetPtr ReadDataset(const std::string& file_path, int64_t& nb) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
}

const std::string base_file = "";
const std::string query_file = "";

int main(int argc, char **argv)
{
    // read data
    // auto train_ds = ReadDataset(base_file);
    auto train_ds = GenDataSet(100000, 128);
    // auto query_ds = ReadDataset(query_file);
    auto query_ds = GenDataSet(100, 128);
    // 1 level kmeans
    float* base_vecs = (float*) train_ds->GetTensor();
    int64_t nb = train_ds->GetRows();
    int64_t dim = train_ds->GetDim();
    float* query_vecs = (float*) query_ds->GetTensor();
    int64_t nq = query_ds->GetRows();

    // for each 1 level, 2 level kmeans
    auto kmeans = knowhere::kmeans::KMeans<float>(nb, dim);
    kmeans.fit(base_vecs, nb);
    // query 1 level, calculate recall

    // query 2 level, calculate recall

    return 0;
}
