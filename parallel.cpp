#include "parallel.h"
#include <list>
#include <thread>
#include <condition_variable>

// From https://github.com/mmp/pbrt-v3/blob/master/src/core/parallel.cpp

static std::vector<std::thread> threads;
static bool shutdownThreads = false;
struct ParallelForLoop;
static ParallelForLoop *workList = nullptr;
static std::mutex workListMutex;
struct ParallelForLoop {
    ParallelForLoop(std::function<void(int64_t)> func1D, int64_t maxIndex, int chunkSize)
        : func1D(std::move(func1D)), maxIndex(maxIndex), chunkSize(chunkSize) {
    }
    ParallelForLoop(const std::function<void(Vector2i)> &f, const Vector2i count)
        : func2D(f), maxIndex(std::get<0>(count) * std::get<1>(count)), chunkSize(1) {
        nX = std::get<0>(count);
    }

    std::function<void(int64_t)> func1D;
    std::function<void(Vector2i)> func2D;
    const int64_t maxIndex;
    const int chunkSize;
    int64_t nextIndex = 0;
    int activeWorkers = 0;
    ParallelForLoop *next = nullptr;
    int nX = -1;

    bool Finished() const {
        return nextIndex >= maxIndex && activeWorkers == 0;
    }
};

static std::condition_variable workListCondition;
static void worker_thread_func(const int tIndex) {
    thread_index = tIndex;
    std::unique_lock<std::mutex> lock(workListMutex);
    while (!shutdownThreads) {
        if (!workList) {
            // Sleep until there are more tasks to run
            workListCondition.wait(lock);
        } else {
            // Get work from _workList_ and run loop iterations
            ParallelForLoop &loop = *workList;

            // Run a chunk of loop iterations for _loop_

            // Find the set of loop iterations to run next
            int64_t indexStart = loop.nextIndex;
            int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);

            // Update _loop_ to reflect iterations this thread will run
            loop.nextIndex = indexEnd;
            if (loop.nextIndex == loop.maxIndex)
                workList = loop.next;
            loop.activeWorkers++;

            // Run loop indices in _[indexStart, indexEnd)_
            lock.unlock();
            for (int64_t index = indexStart; index < indexEnd; ++index) {
                if (loop.func1D) {
                    loop.func1D(index);
                }
                // Handle other types of loops
                else {
                    assert(loop.func2D != nullptr);
                    loop.func2D(make_vector2(int(index % loop.nX),
                                             int(index / loop.nX)));
                }
            }
            lock.lock();

            // Update _loop_ to reflect completion of iterations
            loop.activeWorkers--;
            if (loop.Finished())
                workListCondition.notify_all();
        }
    }
}

void parallel_for(const std::function<void(int64_t)> &func,
                  int64_t count,
                  int chunkSize) {
    // Run iterations immediately if not using threads or if _count_ is small
    if (count < chunkSize) {
        for (int64_t i = 0; i < count; ++i) {
            func(i);
        }
        return;
    }

    // Launch worker threads if needed
    if (threads.size() == 0) {
        thread_index = 0;
        for (int i = 0; i < num_system_cores() - 1; ++i) {
            threads.push_back(std::thread(worker_thread_func, i + 1));
        }
    }

    // Create and enqueue _ParallelForLoop_ for this loop
    ParallelForLoop loop(func, count, chunkSize);
    workListMutex.lock();
    loop.next = workList;
    workList = &loop;
    workListMutex.unlock();

    // Notify worker threads of work to be done
    std::unique_lock<std::mutex> lock(workListMutex);
    workListCondition.notify_all();

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished()) {
        // Run a chunk of loop iterations for _loop_

        // Find the set of loop iterations to run next
        int64_t indexStart = loop.nextIndex;
        int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);

        // Update _loop_ to reflect iterations this thread will run
        loop.nextIndex = indexEnd;
        if (loop.nextIndex == loop.maxIndex) {
            workList = loop.next;
        }
        loop.activeWorkers++;

        // Run loop indices in _[indexStart, indexEnd)_
        lock.unlock();
        for (int64_t index = indexStart; index < indexEnd; ++index) {
            if (loop.func1D) {
                loop.func1D(index);
            }
            // Handle other types of loops
            else {
                assert(loop.func2D != nullptr);
                loop.func2D(make_vector2(int(index % loop.nX),
                                         int(index / loop.nX)));
            }
        }
        lock.lock();

        // Update _loop_ to reflect completion of iterations
        loop.activeWorkers--;
    }
}

thread_local int thread_index;

void parallel_for(std::function<void(Vector2i)> func, const Vector2i count) {
    // Launch worker threads if needed
    if (threads.size() == 0) {
        thread_index = 0;
        for (int i = 0; i < num_system_cores() - 1; ++i)
            threads.push_back(std::thread(worker_thread_func, i + 1));
    }

    ParallelForLoop loop(std::move(func), count);
    {
        std::lock_guard<std::mutex> lock(workListMutex);
        loop.next = workList;
        workList = &loop;
    }

    std::unique_lock<std::mutex> lock(workListMutex);
    workListCondition.notify_all();

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished()) {
        // Run a chunk of loop iterations for _loop_

        // Find the set of loop iterations to run next
        int64_t indexStart = loop.nextIndex;
        int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);

        // Update _loop_ to reflect iterations this thread will run
        loop.nextIndex = indexEnd;
        if (loop.nextIndex == loop.maxIndex) {
            workList = loop.next;
        }
        loop.activeWorkers++;

        // Run loop indices in _[indexStart, indexEnd)_
        lock.unlock();
        for (int64_t index = indexStart; index < indexEnd; ++index) {
            if (loop.func1D) {
                loop.func1D(index);
            }
            // Handle other types of loops
            else {
                assert(loop.func2D != nullptr);
                loop.func2D(make_vector2(int(index % loop.nX),
                                         int(index / loop.nX)));
            }
        }
        lock.lock();

        // Update _loop_ to reflect completion of iterations
        loop.activeWorkers--;
    }
}

int num_system_cores() {
    // return 1;
    int ret = std::thread::hardware_concurrency();
    if (ret == 0) {
        return 16;
    }
    return ret;
}

void terminate_worker_threads() {
    if (threads.size() == 0) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(workListMutex);
        shutdownThreads = true;
        workListCondition.notify_all();
    }

    for (std::thread &thread : threads) {
        thread.join();
    }
    threads.erase(threads.begin(), threads.end());
    shutdownThreads = false;
}
