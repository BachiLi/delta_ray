#pragma once

#include <mutex>
#include <chrono>
#include <ctime>

class ProgressReporter {
    public:
    ProgressReporter(uint64_t total_work, bool print_progress) :
            total_work(total_work), work_done(0), print_progress(print_progress) {
        start_time = std::chrono::system_clock::now();
    }
    void update(uint64_t num) {
        if (print_progress) {
            std::lock_guard<std::mutex> lock(mutex);
            work_done += num;
            float work_ratio = (float)work_done / (float)total_work;
            fprintf(stdout,
                    "\r %.2f Percent Done (%llu / %llu)",
                    work_ratio * float(100.0),
                    (unsigned long long)work_done,
                    (unsigned long long)total_work);
            fflush(stdout);
        }
    }
    void done() {
        if (print_progress) {
            work_done = total_work;
            fprintf(stdout,
                    "\r %.2f Percent Done (%llu / %llu)\n",
                    float(100.0),
                    (unsigned long long)work_done,
                    (unsigned long long)total_work);
            fflush(stdout);
            std::chrono::duration<double> elapsed_seconds =
                std::chrono::system_clock::now() - start_time;
            std::cout << "Elapsed time:" << elapsed_seconds.count() << "s" << std::endl;
        }
    }

    private:
    const uint64_t total_work;
    uint64_t work_done;
    std::mutex mutex;
    std::chrono::system_clock::time_point start_time;
    bool print_progress;
};