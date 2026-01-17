#pragma once

#include <cassert>
#include <thread>
#include <vector>

#define DEFAULT_THREAD_COUNT 4

namespace QDVO::ParallelAlgorithms {

// Finds the number of threads on the user system or returns a default thread
// count.
inline unsigned getSuggestedThreadCount() {
  auto nCores = std::thread::hardware_concurrency();
  if (nCores == 0) {
    return DEFAULT_THREAD_COUNT;
  } else {
    return nCores;
  }
}

// Computes a set of uniformly spaced indices s.t. the first index is 0, the
// last index is (length), and the total number of indices is (nThreads+1).
inline std::vector<int> splitProblem(unsigned nThreads, unsigned length) {
  assert(nThreads > 0);
  std::vector<int> indices(nThreads + 1);

  for (size_t i = 0; i < indices.size(); ++i) {
    indices.at(i) = round((float(i) / nThreads) * (length));
  }

  assert(indices.at(0) == 0);
  assert(indices.back() == length);

  return indices;
}

// Creates a set of upper and lower shifts for each thread to use. Tuple order:
// {lower, upper}
inline std::vector<std::tuple<int, int>> createListDividers(
    unsigned length, int nThreadsOverride = -1) {
  auto nThreads = getSuggestedThreadCount();
  // Allow the number of threads to be overriden.
  if (nThreadsOverride >= 1 && nThreadsOverride < nThreads) {
    nThreads = nThreadsOverride;
  }

  auto splitIndices = splitProblem(nThreads, length);
  std::vector<std::tuple<int, int>> dividers;
  dividers.reserve(nThreads);
  for (int i = 0; i < nThreads; ++i) {
    dividers.push_back(
        std::make_tuple(splitIndices.at(i), splitIndices.at(i + 1)));
  }
  return dividers;
}

inline void joinThreads(std::vector<std::thread> &threads) {
  // Join all threads
  for (auto &th : threads) {
    th.join();
  }
}

}  // namespace QDVO::ParallelAlgorithms
