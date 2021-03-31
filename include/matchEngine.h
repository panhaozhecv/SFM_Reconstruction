//
// Created by panhaozhe on 2021/3/9.
//

#ifndef SFM_MATCHENGINE_H
#define SFM_MATCHENGINE_H

#include "camera.h"
#include "mulThread.h"

class matchEngine {
public:
    typedef std::shared_ptr<matchEngine> Ptr;

    explicit matchEngine(int _mainID, int _numCams):mainID(_mainID), numCams(_numCams){
        p_threadManager = std::make_shared<ThreadManager>(_numCams - 1);
        for(int i = 0; i < _numCams; ++i) {
            for(int j = 0; j < _numCams; ++j) {
                if(j != i) {
                    std::string str = std::to_string(i) + "-" + std::to_string(j);
                    matchList.insert(std::make_pair(str, false));
                }
            }
        }
        lock.unlock();
    };

    ~matchEngine() = default;

    bool is_matched_all_cameras();

    std::pair<int, int> getInitialPair();

    std::queue<int> getAddCamSequence();

    bool sfm_find_initial_pair(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio, const int& thresh);

    void set_initial_pair(const int& i1, const int& i2);

    void sfm_exhaustive_match(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio);


    bool sfm_find_initial_pair_mul_thread(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, std::vector<std::pair<int, int>>& matchedNums, const double& ratio, const int& thresh);

    void sfm_exhaustive_match_mul_thread(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio);

    int update_finished_cams(std::vector<camera::Ptr>& pcams);



    std::unordered_map<std::string, bool> matchList;

private:
    int numCams;
    int mainID;
    std::pair<int, int> initialPair;
    std::set<int> matchedCams;
    std::queue<int> addCamSequence;
    ThreadManager::Ptr p_threadManager;

    std::mutex lock;
};




#endif //SFM_MATCHENGINE_H
