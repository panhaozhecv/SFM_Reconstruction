//
// Created by panhaozhe on 2021/3/13.
//

#ifndef SFM_BAENGINE_H
#define SFM_BAENGINE_H
#include <utility>

#include "camera.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

struct ReprojectCost
{
    cv::Point2f observation;

    ReprojectCost(cv::Point2f& observation)
            : observation(observation)
    {
    }

    template <typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];

        T pos_proj[3];
        ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

        // Apply the camera translation
        pos_proj[0] += t[0];
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];
        const T y = pos_proj[1] / pos_proj[2];

        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];

        // Apply intrinsic
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(observation.x);
        residuals[1] = v - T(observation.y);

        return true;
    }
};

class baEngine {
public:
    typedef std::shared_ptr<baEngine> Ptr;

    baEngine(std::vector<camera::Ptr>& pcameras, cv::Mat K) {
        cv::Mat K1(cv::Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
        intrinsic = K1;
        for(int i = 0; i < pcameras.size(); ++i) {
            cv::Mat temp(6, 1, CV_64FC1);
            if(pcameras[i]->R.data) {
                cv::Mat rvec;
                cv::Rodrigues(pcameras[i]->R, rvec);
                rvec.copyTo(temp.rowRange(0, 3));
                pcameras[i]->T.copyTo(temp.rowRange(3,6));
            }
            else {
                for(int i = 0; i < 6; ++i) {
                    temp.at<double>(i, 0) = 0.0;
                }
            }
            extrinsics.insert(std::make_pair(pcameras[i]->id, temp));
        }
    }

    ~baEngine() = default;

    void BA_config(const int& numThread);

    void BA_run(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& tracks, std::vector<camera::Ptr>& pcameras, const std::set<int>& camMaps);


    void BA_updateExtrinsics(std::vector<camera::Ptr>& pcameras);


    std::unordered_map<int, cv::Mat> extrinsics;
    cv::Mat intrinsic;
    ceres::Solver::Options option;


};


void sfm_reconstruct_BA(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, baEngine::Ptr& p_ba, std::vector<camera::Ptr>& pcameras, const std::set<int>& camIDs);


#endif //SFM_BAENGINE_H
