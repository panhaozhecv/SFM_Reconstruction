//
// Created by panhaozhe on 2021/3/13.
//
#include "baEngine.h"


void sfm_reconstruct_BA(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, baEngine::Ptr& p_ba, std::vector<camera::Ptr>& pcameras, const std::set<int>& camIDs) {
    std::cout << "****************Bundle Adjustment****************" << std::endl;
    p_ba->BA_run(sfmDataMap, sfmDatas, pcameras, camIDs);
}


void baEngine::BA_updateExtrinsics(std::vector<camera::Ptr>& pcameras) {
    for(int i = 0; i < pcameras.size(); ++i) {
        cv::Mat temp(6, 1, CV_64FC1);
        if(pcameras[i]->R.data) {
            cv::Mat rvec;
            cv::Rodrigues(pcameras[i]->R, rvec);
            rvec.copyTo(temp.rowRange(0, 3));
            pcameras[i]->T.copyTo(temp.rowRange(3,6));
        }
        else {
            for(int j = 0; j < 6; ++j) {
                temp.at<double>(j, 0) = 0.0;
            }
        }
        extrinsics[pcameras[i]->id] = temp;
    }
}


void baEngine::BA_config(const int& numThread) {
    option.minimizer_progress_to_stdout = false;
    option.logging_type = ceres::SILENT;
    option.num_threads = numThread;
    option.preconditioner_type = ceres::JACOBI;
    option.linear_solver_type = ceres::SPARSE_SCHUR;
    option.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
}

void baEngine::BA_run(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& tracks, std::vector<camera::Ptr>& pcameras, const std::set<int>& camMaps) {
    ceres::Problem problem;
    // load extrinsics (rotations and motions)
    for(int i = 0; i < pcameras.size(); ++i) {
        if (camMaps.find(pcameras[i]->id) != camMaps.end()) {
            problem.AddParameterBlock(extrinsics[pcameras[i]->id].ptr<double>(), 6);
        }
    }
    // fix the first camera.
    //problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

    // load intrinsic
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy
    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
    std::set<int> foundIDs;
    for(int i = 0; i < pcameras.size(); ++i) {
        if(camMaps.find(pcameras[i]->id) != camMaps.end()) {
            for (auto it = pcameras[i]->matchedPointIDs.begin(); it != pcameras[i]->matchedPointIDs.end(); ++it) {
                for(int pointIndex = 0;pointIndex < it->second.size(); ++ pointIndex) {
                    int camID = pcameras[i]->id;
                    cv::Point2f observed = pcameras[i]->features[it->second[pointIndex]].pt;
                    std::string searchStr = std::to_string(observed.x) + "," + std::to_string(observed.y);
                    int sfmDataId = search_sfm_data(sfmDataMap, searchStr, camID);
                    if(sfmDataId != -1) {
                        if(tracks[sfmDataId].isInlier && foundIDs.find(sfmDataId) == foundIDs.end()) {
                            foundIDs.insert(sfmDataId);
                            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(
                                    new ReprojectCost(observed));
                            problem.AddResidualBlock(
                                    cost_function,
                                    loss_function,
                                    intrinsic.ptr<double>(),
                                    extrinsics[pcameras[i]->id].ptr<double>(),
                                    &(tracks[sfmDataId].pos[0])
                            );
                        }
                    }
                }
            }
        }
    }
    ceres::Solver::Summary summary;
    ceres::Solve(option, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        std::cout << std::endl
                  << "Bundle Adjustment statistics (approximated RMSE):\n"
                  << " #views: " << pcameras.size() << "\n"
                  << " #residuals: " << summary.num_residuals << "\n"
                  << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
                  << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
                  << " Time (s): " << summary.total_time_in_seconds << "\n"
                  << std::endl;
    }
    for(int i = 0; i < pcameras.size(); ++i) {
        if(camMaps.find(i) != camMaps.end()) {
            cv::Mat refinedE = extrinsics[pcameras[i]->id];
            cv::Mat rvec(3, 1, CV_64FC1);
            cv::Mat tvec(3, 1, CV_64FC1);
            rvec = refinedE.rowRange(0, 3);
            tvec = refinedE.rowRange(3, 6);
            cv::Mat rotation;
            cv::Rodrigues(rvec, rotation);
            pcameras[i]->setExtrinsic(rotation, tvec);
        }
    }
}
