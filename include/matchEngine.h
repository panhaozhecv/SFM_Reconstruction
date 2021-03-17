//
// Created by panhaozhe on 2021/3/9.
//

#ifndef SFM_MATCHENGINE_H
#define SFM_MATCHENGINE_H

#include "camera.h"

class matchEngine {
public:
    typedef std::shared_ptr<matchEngine> Ptr;

    explicit matchEngine(int _mainID, int _numCams):mainID(_mainID), numCams(_numCams){};

    ~matchEngine() = default;

    bool is_matched_all_cameras();

    std::pair<int, int> getInitialPair();

    std::queue<int> getAddCamSequence();

    bool sfm_find_initial_pair(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio, const int& thresh);

    void set_initial_pair(const int& i1, const int& i2);

    void sfm_exhaustive_match(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, std::vector<camera::Ptr>& p_cams, const double& ratio);
private:
    int numCams;
    int mainID;
    std::pair<int, int> initialPair;
    std::set<int> matchedCams;
    std::queue<int> addCamSequence;


};




#endif //SFM_MATCHENGINE_H
