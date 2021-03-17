#include <iostream>
#include <vector>
#include <memory>
#include "cpptoml.h"
#include "camera.h"
#include "mulThread.h"
#include "matchEngine.h"
#include "baEngine.h"

auto tomlConfig = cpptoml::parse_file("../config/config.toml");
std::string projectPath; 
std::string picPath;
std::string featPath;
std::string plyFilePath;
int camNum(0);
int mainCamID = *tomlConfig->get_qualified_as<int>("mainCamID");

int main() {
    projectPath = *tomlConfig->get_qualified_as<std::string>("project_folder");
    picPath = projectPath + "/color";
    featPath = picPath + "/feature";
    plyFilePath = projectPath + "/out.ply";
    camNum = *tomlConfig->get_qualified_as<int>("camNum");

    ThreadManager::Ptr p_threadManager = std::make_shared<ThreadManager>(camNum);


    double fx = *tomlConfig->get_qualified_as<double>("fx");
    double fy = *tomlConfig->get_qualified_as<double>("fy");
    double cx = *tomlConfig->get_qualified_as<double>("cx");
    double cy = *tomlConfig->get_qualified_as<double>("cy");

    double k1 = *tomlConfig->get_qualified_as<double>("k1");
    double k2 = *tomlConfig->get_qualified_as<double>("k2");
    double p1 = *tomlConfig->get_qualified_as<double>("p1");
    double p2 = *tomlConfig->get_qualified_as<double>("p2");

    std::vector<sfmData> sfmDatas;
    std::multimap<std::string, cv::Vec2i> sfmDataMap;
    std::vector<camera::Ptr> p_cams;


    // 初始化 绑定内参及图片
    std::cout << "****************Initialize Cameras****************"  << std::endl;
    for(int i = 0; i < camNum; ++i) {
        char tmpPath[50];
        sprintf(tmpPath, "/%04d.jpg", i+1);
        std::string imgPath = picPath + std::string(tmpPath);
        std::cout << "Init View:" <<  imgPath << std::endl;
        cv::Mat tmpImg = cv::imread(imgPath);
        camera::Ptr p_temp = std::make_shared<camera>(i, tmpImg);
        p_temp->setIntrinsic(fx, fy, cx, cy);
        p_temp->setDistCoeffs(k1, k2, p1, p2);
        p_cams.push_back(p_temp);
    }
    // 特征提取与描述子

    std::cout << "****************Extract Features****************"  << std::endl;
    for(int i = 0; i < camNum; ++i) {
        p_cams[i]->sfm_extract_feature_desc(0.02, 10);
    }
    // 特征匹配
    matchEngine::Ptr p_matcher = std::make_shared<matchEngine>(mainCamID, camNum);

    bool isInit = p_matcher->sfm_find_initial_pair(sfmDataMap, sfmDatas, p_cams, 0.6, 100);
    if(!isInit) {
        return -1;
    }
    p_matcher->sfm_exhaustive_match(sfmDataMap, sfmDatas, p_cams, 0.4);


    // 计算RT 重建
    std::queue<int> addCamsSeq = p_matcher->getAddCamSequence();
    int secCam = p_matcher->getInitialPair().second;
    // 初始匹配对重建
    int numInliers(0);
    sfm_reconstruct_initial_pair(sfmDataMap, sfmDatas, numInliers, p_cams[mainCamID], p_cams[secCam]);
    // BA

    baEngine::Ptr p_ba = std::make_shared<baEngine>(p_cams, p_cams[mainCamID]->K);
    p_ba->BA_config(4);

    std::set<int> BA_cams = {mainCamID, secCam};
    sfm_reconstruct_BA(sfmDataMap, sfmDatas, p_ba, p_cams, BA_cams);

    // 增量式重建
    std::set<int> finishedCamIDs = {mainCamID, secCam};
    while(!addCamsSeq.empty()) {
       int addCamID = addCamsSeq.front();
       BA_cams.insert(addCamID);
       addCamsSeq.pop();
       sfm_incremental_reconstruction(sfmDataMap, sfmDatas, numInliers, finishedCamIDs, p_cams, addCamID);
       finishedCamIDs.insert(addCamID);
       p_ba->BA_updateExtrinsics(p_cams);
       sfm_reconstruct_BA(sfmDataMap, sfmDatas, p_ba, p_cams, BA_cams);

    }


    sfm_draw_cam_position(sfmDatas, p_cams, numInliers, mainCamID, cv::Vec3b(0,255,0));
    sfm_write_data_to_ply(sfmDatas, numInliers, plyFilePath);
    std::string cmd = "../tool/meshlab.AppImage " + plyFilePath;
    std::system(cmd.c_str());
    return 0;
}