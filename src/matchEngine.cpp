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
                int numMatch = sfm_match_feature( sfmDataMap, sfmDatas, p_cams[currID], p_cams[i], ratio);
                std::cout << "View " << currID << "-" << i << " Matched " << numMatch << " Points" << std::endl;
                if (numMatch >= numMax) {
                    numMax = numMatch;
                    bestID = i;
                }
            }
        }
        matchedCams.insert(currID);
        currID = bestID;
        if(addCamSequence.size() < numCams-2) {
            addCamSequence.push(currID);
        }
    }
}
