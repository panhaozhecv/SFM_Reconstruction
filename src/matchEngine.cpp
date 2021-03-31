//
// Created by panhaozhe on 2021/3/9.
//
#include "matchEngine.h"


void matchEngine::set_initial_pair(const int& i1, const int& i2) {
    initialPair = std::make_pair(i1, i2);
}

std::queue<int> matchEngine::getAddCamSequence() {
    return addCamSequence;
}

std::pair<int, int> matchEngine::getInitialPair() {
    return initialPair;
}


bool matchEngine::is_matched_all_cameras() {
    for(int i = 0; i < numCams; ++i) {
        if(matchedCams.find(i) == matchedCams.end()) {
            return false;
        }
    }
    return true;
}

bool matchEngine::sfm_find_initial_pair(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio, const int& thresh) {
    std::cout << "****************Find Initial Pair****************"  << std::endl;
    int numMax(0);
    int bestID(mainID);
    for(int i = 0; i < p_cams.size(); ++i) {
        if(i != mainID) {
            std::pair<int, int> matchNum = std::make_pair(p_cams[i]->id, 0);
            int numMatch = sfm_match_feature(sfmDataMap, sfmDatas, p_cams[mainID], p_cams[i], ratio);
            std::cout << "View " << mainID << "-" << i << " Matched " << numMatch << " Points" << std::endl;
            if (numMatch >= numMax) {
                numMax = numMatch;
                bestID = i;
            }
        }
    }
    if(numMax < thresh) {
        std::cout << "Cannot Find Good Initial Pair!" << std::endl;
        return false;
    }
    set_initial_pair(mainID, bestID);
    matchedCams.insert(mainID);
    matchedCams.insert(bestID);
    std::cout << "Find Initial Pair <" << mainID << "," << bestID << ">" << std::endl;
    return true;
}

bool matchEngine::sfm_find_initial_pair_mul_thread(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, std::vector<std::pair<int, int>>& matchedNums, const double& ratio, const int& thresh) {
    std::cout << "****************Find Initial Pair****************"  << std::endl;
    for(int i = 0; i < p_cams.size(); ++i) {
        if(p_cams[i]->id != mainID) {
            p_threadManager->addFunction(sfm_match_feature_mul_thread, std::ref(sfmDataMap), \
                                         std::ref(sfmDatas), std::ref(matchList), std::ref(p_cams[mainID]), \
                                         std::ref(p_cams[i]), std::ref(matchedNums[i]), \
                                         std::ref(lock), ratio);
        }

    }
    p_threadManager->runFunction();
    int numMax(0);
    int bestID(mainID);
    for(auto it : matchedNums) {
        if(it.second >= numMax) {
            numMax = it.second;
            bestID = it.first;
        }
    }
    if(numMax < thresh) {
        std::cout << "Cannot Find Good Initial Pair!" << std::endl;
        return false;
    }
    set_initial_pair(mainID, bestID);
    matchedCams.insert(mainID);
    matchedCams.insert(bestID);
    std::cout << "Find Initial Pair <" << mainID << "," << bestID << ">" << std::endl;
    lock.unlock();
    return true;
}




void matchEngine::sfm_exhaustive_match(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio) {
    std::cout << "****************Exhaustive Match Begin****************"  << std::endl;
    int currID(initialPair.second);
    int bestID(initialPair.second);
    int numMax(0);
    while(!is_matched_all_cameras()) {
        numMax = 0;
        bestID = currID;
        for(int i = 0; i < p_cams.size(); ++i) {
            if(i != currID && matchedCams.find(i) == matchedCams.end()) {
                //std::cout << currID << " " << bestID << std::endl;
                std::pair<int, int> matchNum = std::make_pair(p_cams[i]->id, 0);
                int numMatch = sfm_match_feature( sfmDataMap, sfmDatas, p_cams[currID], p_cams[i], ratio);
                std::cout << "View " << currID << "-" << i << " Matched " << numMatch << " Points" << std::endl;
                if (numMatch >= numMax) {
                    numMax = numMatch;
                    bestID = i;
                }
            }
        }
        p_cams[currID]->isFinishedMatch = true;
        matchedCams.insert(currID);
        currID = bestID;
        if(addCamSequence.size() < numCams-2) {
            addCamSequence.push(currID);
        }
    }
}

int matchEngine::update_finished_cams(std::vector<camera::Ptr>& pcams) {
    for(int i = 0; i < pcams.size(); ++i) {
        if(matchedCams.find(pcams[i]->id) == matchedCams.end() && pcams[i]->isFinishedMatch) {
            matchedCams.insert(pcams[i]->id);
        }
    }
    return static_cast<int>(matchedCams.size());
}


void matchEngine::sfm_exhaustive_match_mul_thread(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio) {
    std::cout << "****************Exhaustive Match Begin****************"  << std::endl;
    while(!is_matched_all_cameras()) {
        for(int i = 0; i < p_cams.size(); ++i) {
            if(p_cams[i]->id != mainID && !p_cams[i]->isFinishedMatch) {
                p_threadManager->addFunction(sfm_one_cam_match, std::ref(sfmDataMap), std::ref(sfmDatas), \
                                             std::ref(matchList), std::ref(p_cams), std::ref(lock), i, ratio);
            }
        }
        p_threadManager->runFunction();
        int numLeft = numCams - update_finished_cams(p_cams);
        p_threadManager->refresh(numLeft);
    }
}