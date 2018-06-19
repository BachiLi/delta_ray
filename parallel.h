#pragma once

#include "vector.h"

#include <mutex>
#include <functional>
#include <atomic>

void parallel_for(const std::function<void(int64_t)> &func,
                  int64_t count,
                  int chunkSize = 1);
extern thread_local int thread_index;
void parallel_for(std::function<void(Vector2i)> func, const Vector2i count);
int num_system_cores();
void terminate_worker_threads();