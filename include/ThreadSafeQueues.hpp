#ifndef THREAD_SAFE_QUEUES_HPP
#define THREAD_SAFE_QUEUES_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * @file ThreadSafeQueues.hpp
 * @brief 线程安全队列模板库，包含无界队列（UnboundedSafeQueue）和有界队列（BoundedSafeQueue）。
 */

// ==============================================
// 无界线程安全队列（内存无限制）
// ==============================================
template <typename T>
class UnboundedSafeQueue {
public:
    UnboundedSafeQueue() = default;

    // 入队（永远成功，除非内存耗尽）
    void enqueue(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_not_empty_.notify_one();
    }

    // 出队（队列空时阻塞）
    bool dequeue(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // 标记队列终止
    void setFinished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_not_empty_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::atomic<bool> finished_{false};
};

// ==============================================
// 有界线程安全队列（固定容量）
// ==============================================
template <typename T>
class BoundedSafeQueue {
public:
    explicit BoundedSafeQueue(size_t max_size) : max_size_(max_size) {}

    // 入队（队列满时阻塞）
    bool enqueue(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this] { return queue_.size() < max_size_ || finished_; });
        if (finished_) return false;
        queue_.push(std::move(item));
        cv_not_empty_.notify_one();
        return true;
    }

    // 出队（队列空时阻塞）
    bool dequeue(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    // 标记队列终止
    void setFinished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    std::queue<T> queue_;
    const size_t max_size_;
    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::atomic<bool> finished_{false};
};

#endif // THREAD_SAFE_QUEUES_HPP