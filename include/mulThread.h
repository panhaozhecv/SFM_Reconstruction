//
// Created by panhaozhe on 2021/3/9.
//

#ifndef SFM_MULTHREAD_H
#define SFM_MULTHREAD_H

#include <vector>
#include <iostream>
#include <thread>
#include <memory>
#include <queue>

class ThreadManager {

public:
    typedef std::shared_ptr<ThreadManager> Ptr;
    ThreadManager(const int& _numTasks):numTasks(_numTasks) {
        numCores = static_cast<int>(std::thread::hardware_concurrency());
    }

    ~ThreadManager() = default;

    template<class F, class... Args>
    void addFunction(F& f, Args&&... args) {
        auto* t = new std::thread(f, args...);
        taskQueue.push(t);
    }


    void runFunction();

private:
    int numCores;
    int numTasks;
    std::queue<std::thread*> taskQueue;

};






#endif //SFM_MULTHREAD_H
