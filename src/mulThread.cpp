//
// Created by panhaozhe on 2021/3/9.
//
#include "mulThread.h"

void  ThreadManager::runFunction() {
    int numFinished(0);
    std::vector<std::thread*> taskPool;
    while(numFinished <= numTasks) {
        while(numFinished <= numTasks && taskPool.size() <= numCores) {
            taskPool.push_back(taskQueue.front());
            taskQueue.pop();
        }
        for(auto t : taskPool) {
            t->join();
            numFinished += 1;
        }
        taskPool.clear();
        while(!taskQueue.empty())
            taskQueue.pop();
    }

}

