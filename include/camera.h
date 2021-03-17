#ifndef SFM_CAMERA_H
#define SFM_CAMERA_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/xfeatures2d/cuda.hpp>


struct sfmData {
    std::unordered_map<int, cv::Point2f> pixelPoints;
    std::set<int> containedViews;

    bool isInlier;
    cv::Vec3d pos;
    cv::Vec3b color;
};



class camera {
    public:
        typedef std::shared_ptr<camera> Ptr;

        camera(int _id, cv::Mat& _img):
            id(_id),
            img(_img){}

        ~camera() = default;

        // 设置内参
        void setIntrinsic(const double& fx, const double& fy, const double& cx, const double& cy);

        // 畸变参数
        void setDistCoeffs(const double& k1, const double& k2, const double& p1, const double& p2);

        // 设置外参
        void setExtrinsic(const cv::Mat& _R, const cv::Mat& _T);

        // 提取特征 生成描述子
        void sfm_extract_feature_desc(double contrastThreshold = 0.04, double edgeThreshold = 10);

        void sfm_extract_feature_desc_gpu();

        void test();

        std::vector<cv::KeyPoint> features;
        cv::Mat descs;
        int numPtsAll;
        int numPtsMatched;
        std::unordered_map<int, int> matchMap;
        std::unordered_map<int, std::vector<int>> matchedPointIDs;
        std::unordered_map<int, cv::Mat> inlierIDs;


        int id;
        cv::Mat img;
        cv::Mat K;
        cv::Mat distCoeffs;

        cv::Mat R;
        cv::Mat T;
        cv::Mat Extrinsic;



};

void thread_extract_feature_desc(camera::Ptr& p_cam, double contrastThreshold = 0.04, double edgeThreshold = 10);

// 特征匹配
int sfm_match_feature(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, camera::Ptr& src, camera::Ptr& dst,const double& ratio);

// sfm_data添加映射二维点
void sfm_data_add_2dPoint(sfmData& data, const int& id, const cv::Point2f& point);

// 计算位姿
void sfm_reconstruct_initial_pair(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, int& numInliers, camera::Ptr& src, camera::Ptr& dst);

// 增量重建
void sfm_incremental_reconstruction(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, int& numInliers, const std::set<int>& finishedCamIDs, std::vector<camera::Ptr>& pcams, const int& index);

void sfm_draw_cam_position(std::vector<sfmData>& datas, std::vector<camera::Ptr>& pcams, int& numVertex, const int& mainCamID, const cv::Vec3b& color);

void sfm_write_data_to_ply(std::vector<sfmData>& datas, const int& numVertex, const std::string& filePath);

int search_sfm_data(const std::multimap<std::string, cv::Vec2i>& sfmDataMap, const std::string& point, const int& camID);

void sfm_reproj_3dpoints(camera::Ptr& pcam1, camera::Ptr& pcam2, const std::vector<cv::Point3f>& points);

#endif
