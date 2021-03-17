//
// Created by panhaozhe on 2021/3/8.
//
#include "camera.h"


void thread_extract_feature_desc(camera::Ptr &p_cam, double contrastThreshold, double edgeThreshold) {
    //p_cam->sfm_extract_feature_desc(contrastThreshold, edgeThreshold);
    p_cam->test();
}

void sfm_reproj_3dpoints(camera::Ptr& pcam1, camera::Ptr& pcam2, const std::vector<cv::Point3f>& points) {
    cv::Mat img1 = pcam1->img.clone();
    cv::Mat img2 = pcam2->img.clone();
    std::vector<cv::Point2f> src2dpts, dst2dpts;
    cv::Mat rVec(3, 1, cv::DataType<float>::type); // Rotation vector
    rVec.at<float>(0) = 0.0;
    rVec.at<float>(1) = 0.0;
    rVec.at<float>(2) = 0.0;
    cv::Mat tVec(3, 1, cv::DataType<float>::type); // Translation vector
    tVec.at<double>(0) = 0.0;
    tVec.at<double>(1) = 0.0;
    tVec.at<double>(2) = 0.0;
    cv::Rodrigues(pcam1->R, rVec);
    tVec = pcam1->T;
    cv::projectPoints(points, rVec, tVec, pcam1->K, cv::noArray(), src2dpts);
    cv::Rodrigues(pcam2->R, rVec);
    tVec = pcam2->T;
    cv::projectPoints(points, rVec, tVec, pcam2->K, cv::noArray(), dst2dpts);

    for(int i = 0; i < src2dpts.size(); ++i) {
        cv::circle(img1, src2dpts[i], 15.0, cv::Scalar(0, 0, 255));
        cv::circle(img2, dst2dpts[i], 15.0, cv::Scalar(0, 0, 255));
    }
    cv::Mat test1;
    cv::hconcat(img1, img2, test1);
    cv::namedWindow("reproj", cv::WINDOW_NORMAL);
    cv::imshow("reproj", test1);
    cv::waitKey();
    cv::destroyAllWindows();
}


int search_sfm_data(const std::multimap<std::string, cv::Vec2i>& sfmDataMap, const std::string& pointStr, const int& camID) {
    auto iter = sfmDataMap.find(pointStr);
    if(iter == sfmDataMap.end()) {
        return -1;
    }
    for(int i = 0; i != sfmDataMap.count(pointStr); i++, iter++) {
        if(iter->second[0] == camID) {
            return iter->second[1];
        }
    }
    return -1;
}



void sfm_data_add_2dPoint(sfmData& data, const int& id, const cv::Point2f& point) {
    if(data.containedViews.find(id) == data.containedViews.end()) {
        data.containedViews.insert(id);
        data.pixelPoints.insert(std::make_pair(id, point));
    }
}

int sfm_match_feature(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, camera::Ptr& src, camera::Ptr& dst,const double& ratio) {
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> matchedPoints;
    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Mat> train_desc(1, dst->descs);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(src->descs, matchedPoints, 2);
    for(int i = 0; i < matchedPoints.size(); ++i) {
        if(matchedPoints[i][0].distance < ratio * matchedPoints[i][1].distance) {
            goodMatches.push_back(matchedPoints[i][0]);
        }
    }

    src->matchMap.insert(std::make_pair(dst->id, goodMatches.size()));
    dst->matchMap.insert(std::make_pair(src->id, goodMatches.size()));
    int numCommon(0);
    int numAll(0);
    std::vector<int> srcMatchedIDs, dstMatchedIDs;
    for(auto it : goodMatches) {
        numAll += 1;
        cv::Point2f srcPts = src->features[it.queryIdx].pt;
        cv::Point2f dstPts = dst->features[it.trainIdx].pt;

            srcMatchedIDs.push_back(it.queryIdx);
            dstMatchedIDs.push_back(it.trainIdx);

            bool shouldAdd = true;
            for (int i = 0; i < sfmDatas.size(); ++i) {
                if (sfmDatas[i].pixelPoints.find(src->id) != sfmDatas[i].pixelPoints.end()) {
                    if (sfmDatas[i].pixelPoints[src->id] == srcPts) {
                        numCommon += 1;
                        sfm_data_add_2dPoint(sfmDatas[i], dst->id, dstPts);

                        std::string dstPtsStr = std::to_string(dstPts.x) + "," + std::to_string(dstPts.y);
                        sfmDataMap.insert(std::make_pair(dstPtsStr, cv::Vec2i(dst->id, i)));
                        shouldAdd = false;
                    }
                }
            }
            if (shouldAdd) {
                sfmData data;

                data.pos = cv::Vec3d(0.0, 0.0, 0.0);
                data.isInlier = false;
                sfm_data_add_2dPoint(data, src->id, srcPts);
                sfm_data_add_2dPoint(data, dst->id, dstPts);
                sfmDatas.push_back(data);

                std::string dstPtsStr = std::to_string(dstPts.x) + "," + std::to_string(dstPts.y);
                std::string srcPtsStr = std::to_string(srcPts.x) + "," + std::to_string(srcPts.y);

                sfmDataMap.insert(std::make_pair(srcPtsStr, cv::Vec2i(src->id, sfmDatas.size() - 1)));
                sfmDataMap.insert(std::make_pair(dstPtsStr, cv::Vec2i(dst->id, sfmDatas.size() - 1)));

            }
    }
    src->matchedPointIDs.insert(std::make_pair(dst->id, srcMatchedIDs));
    dst->matchedPointIDs.insert(std::make_pair(src->id, dstMatchedIDs));
    return goodMatches.size();

}


void sfm_reconstruct_initial_pair(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, int& numInliers, camera::Ptr& src, camera::Ptr& dst) {
    std::cout << "****************Reconstruct Initial Pair****************" << std::endl;
    std::vector<cv::Point2f> srcPts, dstPts;
    std::vector<cv::Point2f> srcPtsInlier, dstPtsInlier;
    std::vector<cv::Point2f> srcPtsUndist, dstPtsUndist;
    for(int i = 0; i < src->matchedPointIDs[dst->id].size(); ++i) {
        srcPts.push_back(src->features[src->matchedPointIDs[dst->id][i]].pt);
    }
    for(int i = 0; i < dst->matchedPointIDs[src->id].size(); ++i) {
        dstPts.push_back(dst->features[dst->matchedPointIDs[src->id][i]].pt);
    }
    cv::Mat inliers;
    cv::Mat essential = cv::findEssentialMat(srcPts, dstPts, dst->K, cv::RANSAC, 0.999, 1.0, inliers);
    cv::Mat R, T;
    cv::recoverPose(essential, srcPts, dstPts, dst->K, R, T, inliers);

    src->inlierIDs.insert(std::make_pair(dst->id, inliers));
    dst->inlierIDs.insert(std::make_pair(src->id, inliers));
    for(int i = 0; i < inliers.rows; ++i) {
        if(inliers.at<uchar>(i)) {
            srcPtsInlier.push_back(srcPts[i]);
            dstPtsInlier.push_back(dstPts[i]);
        }
    }

    cv::undistortPoints(srcPtsInlier, srcPtsUndist, src->K, src->distCoeffs, cv::noArray(), src->K);
    cv::undistortPoints(dstPtsInlier, dstPtsUndist, dst->K, dst->distCoeffs, cv::noArray(), dst->K);



    cv::Mat R1 = cv::Mat::eye(3, 3, R.type());
    cv::Mat T1 = cv::Mat::zeros(3, 1, T.type());
    src->setExtrinsic(R1, T1);

    cv::Mat R2 = R.inv();
    cv::Mat T2 = -R2 * T;
    dst->setExtrinsic(R, T);

    cv::Mat pts_4d;
    cv::Mat proj1(3, 4, CV_64F);
    cv::Mat proj2(3, 4, CV_64F);
    proj1 = src->K * src->Extrinsic;
    proj2 = dst->K * dst->Extrinsic;
    cv::triangulatePoints(proj1, proj2, srcPtsUndist, dstPtsUndist, pts_4d);

    std::vector<cv::Point3f> pos3d;
    for(int i = 0; i < pts_4d.cols; ++i) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3f temp(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        pos3d.push_back(temp);
    }

    cv::Mat final;
    int temp(0);
    for(int i = 0; i < srcPtsUndist.size(); ++i) {
        cv::Vec2d pt1(srcPtsInlier[i].x, srcPtsInlier[i].y);
        cv::Vec2d pt2(dstPtsInlier[i].x, dstPtsInlier[i].y);

        cv::Vec3d pos = cv::Vec3d(pos3d[i].x, pos3d[i].y, pos3d[i].z);

        std::string pt1Str = std::to_string(srcPtsInlier[i].x) + "," + std::to_string(srcPtsInlier[i].y);
        std::string pt2Str = std::to_string(dstPtsInlier[i].x) + "," + std::to_string(dstPtsInlier[i].y);
        int sfmDataId = search_sfm_data(sfmDataMap, pt1Str, src->id);

        if(sfmDataId != -1) {

            if(!sfmDatas[sfmDataId].isInlier) {

                sfmDatas[sfmDataId].pos = pos;
                numInliers += 1;
                sfmDatas[sfmDataId].isInlier = true;
                cv::Point2f pixelCool = sfmDatas[sfmDataId].pixelPoints[src->id];
                cv::Point2f pixelCool1 = sfmDatas[sfmDataId].pixelPoints[dst->id];
                int x = (int)pixelCool.x;
                int y = (int)pixelCool.y;
                int x1 = (int)pixelCool1.x;
                int y1 = (int)pixelCool1.y;

                uchar *ptr = src->img.ptr<uchar>(y);
                uchar b = ptr[3*x];
                uchar g = ptr[3*x+1];
                uchar r = ptr[3*x+2];

                sfmDatas[sfmDataId].color = cv::Vec3b(b, g, r);


            }


        }
    }

}


void sfm_incremental_reconstruction(std::multimap<std::string, cv::Vec2i>& sfmDataMap, std::vector<sfmData>& sfmDatas, int& numInliers, const std::set<int>& finishedCamIDs, std::vector<camera::Ptr>& pcams, const int& index) {
    std::cout << "****************Incremental Reconstruction #Cam[" << pcams[index]->id << "]****************" << std::endl;
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::set<int> foundIDs;
    for(int i = 0; i < finishedCamIDs.size(); ++i) {
        int searchCamID = i;
        for(int j = 0; j < pcams[index]->matchedPointIDs[searchCamID].size(); ++j) {
            cv::Point2f searchPoint = pcams[index]->features[pcams[index]->matchedPointIDs[searchCamID][j]].pt;
            std::string searchStr = std::to_string(searchPoint.x) + "," + std::to_string(searchPoint.y);
            int sfmDataId = search_sfm_data(sfmDataMap, searchStr, pcams[index]->id);
            if(sfmDataId != -1) {
                if(sfmDatas[sfmDataId].isInlier && foundIDs.find(sfmDataId) == foundIDs.end()) {
                    pts3d.push_back(cv::Point3f(sfmDatas[sfmDataId].pos));
                    pts2d.push_back(searchPoint);
                    foundIDs.insert(sfmDataId);
                }
            }
        }
    }
    if(pts3d.size() <= 20) {
        return;
    }
    std::cout << "Cam[" << pcams[index]->id << "] Found " << pts3d.size() << " points for solvePnp" << std::endl;
    cv::Mat rvec, tvec;
    cv::Mat inliers;
    cv::solvePnPRansac(pts3d, pts2d, pcams[index]->K, cv::noArray(), rvec, tvec, false, 500, 6.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    pcams[index]->setExtrinsic(rotation, tvec);

    for(int i = 0; i < pcams.size(); ++i) {
        if(finishedCamIDs.find(i) != finishedCamIDs.end()) {
            std::vector<cv::Point2f> pts1, pts2;
            std::vector<int> pts1ID = pcams[i]->matchedPointIDs[pcams[index]->id];
            std::vector<int> pts2ID = pcams[index]->matchedPointIDs[pcams[i]->id];
            for(int i1 = 0; i1 < pts1ID.size(); ++i1) {
                pts1.push_back(pcams[i]->features[pts1ID[i1]].pt);
                pts2.push_back(pcams[index]->features[pts2ID[i1]].pt);
            }

            if(pts1.size() == 0) {
                continue;
            }
            cv::Mat pts_4d;
            cv::Mat proj1(3, 4, CV_64F);
            cv::Mat proj2(3, 4, CV_64F);
            proj1 = pcams[i]->K * pcams[i]->Extrinsic;
            proj2 = pcams[index]->K * pcams[index]->Extrinsic;
            cv::triangulatePoints(proj1, proj2, pts1, pts2, pts_4d);
            std::vector<cv::Point3f> pos3d;
            for(int i2 = 0; i2 < pts_4d.cols; ++i2) {
                cv::Mat x = pts_4d.col(i2);
                x /= x.at<float>(3, 0);
                cv::Point3f temp(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
                pos3d.push_back(temp);
            }
            // 重投影测试
            //sfm_reproj_3dpoints(pcams[i], pcams[index], pos3d);

            cv::Mat tmpInliers;
            cv::Mat essential = cv::findEssentialMat(pts1, pts2, pcams[i]->K, cv::RANSAC, 0.999, 1.0, tmpInliers);
            pcams[i]->inlierIDs.insert(std::make_pair(pcams[index]->id, tmpInliers));
            pcams[index]->inlierIDs.insert(std::make_pair(pcams[i]->id, tmpInliers));

            int numAdded(0);
            for(int i3 = 0; i3 < pts1.size(); ++i3) {
                if(tmpInliers.at<char>(i3)) {
                    std::string searchStr = std::to_string(pts1[i3].x) + "," + std::to_string(pts1[i3].y);
                    int sfmDataId = search_sfm_data(sfmDataMap, searchStr, pcams[i]->id);
                    if (sfmDataId != -1) {
                        if (!sfmDatas[sfmDataId].isInlier) {
                            sfmDatas[sfmDataId].pos = cv::Vec3d(pos3d[i3].x, pos3d[i3].y, pos3d[i3].z);
                            numInliers += 1;
                            sfmDatas[sfmDataId].isInlier = true;
                            cv::Point2f pixelCool = sfmDatas[sfmDataId].pixelPoints[pcams[i]->id];
                            int x = (int) pixelCool.x;
                            int y = (int) pixelCool.y;
                            uchar *ptr = pcams[i]->img.ptr<uchar>(y);
                            uchar b = ptr[3 * x];
                            uchar g = ptr[3 * x + 1];
                            uchar r = ptr[3 * x + 2];
                            sfmDatas[sfmDataId].color = cv::Vec3b(b, g, r);
                            numAdded += 1;
                        }
                    }
                }
            }
            std::cout << "Cam[" << pcams[i]->id << "-" << pcams[index]->id << "] Added " << numAdded << " tracks" << std::endl;
        }
    }

}

void camera::setIntrinsic(const double& fx, const double& fy, const double& cx, const double& cy) {
    K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                   0.0, fy, cy,
                                   0.0, 0.0, 1.0);
}

void camera::setDistCoeffs(const double& k1, const double& k2, const double& p1, const double& p2) {
    distCoeffs = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);
}


void camera::setExtrinsic(const cv::Mat& _R, const cv::Mat& _T) {
    R = _R;
    T = _T;
    cv::Mat out(3, 4, CV_64F);
    R.copyTo(out(cv::Rect(0,0,3,3)));
    T.copyTo(out.colRange(3, 4));
    Extrinsic = out;
    std::cout << "Cam[" << id << "] Extrinsic:\n" << Extrinsic << std::endl;
}


void camera::sfm_extract_feature_desc(double contrastThreshold, double edgeThreshold) {
//    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create(0, 3,\
                                                                    contrastThreshold, edgeThreshold,1.6);
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();
    detector->detectAndCompute(img, cv::noArray(), features, descs);
    numPtsAll = features.size();
    std::cout << "camera " << id << " extracted " << numPtsAll << " SIFT pts" << std::endl;
}


void camera::sfm_extract_feature_desc_gpu() {

}



void sfm_draw_cam_position(std::vector<sfmData>& datas, std::vector<camera::Ptr>& pcams, int& numVertex, const int& mainCamID, const cv::Vec3b& color) {
    for(int i = 0; i < pcams.size(); ++i) {
        cv::Mat R = pcams[i]->R;
        cv::Mat T = pcams[i]->T;
        cv::Mat R1 = R.inv();
        cv::Mat T1 = -R.inv() * T;
        sfmData temp;
        temp.isInlier = true;
        temp.pos = cv::Vec3d(T1.at<double>(0,0), T1.at<double>(1,0), T1.at<double>(2,0));
        cv::Vec3b tmpColor;
        if(i == mainCamID) {
            tmpColor = cv::Vec3b(0,0,255);
        }
        else {
            tmpColor = color;
        }
        numVertex += 1;
        temp.color = tmpColor;
        datas.push_back(temp);
        for(int j = 1; j <= 10; ++j) {
            cv::Mat srcPos = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.1 * j);
            cv::Mat dstPos = R1 * srcPos + T1;
            sfmData temp;
            temp.isInlier = true;
            temp.pos = cv::Vec3d(dstPos.at<double>(0,0), dstPos.at<double>(1,0), dstPos.at<double>(2,0));
            temp.color = tmpColor;
            datas.push_back(temp);
            numVertex += 1;
        }
    }
}


void sfm_write_data_to_ply(std::vector<sfmData>& datas, const int& numVertex, const std::string& filePath) {
    std::ofstream file(filePath, std::ios::out);
    file << "ply\n"
         << "format ascii 1.0\n"
         << "element vertex " << numVertex << "\n"
         << "property double x\n"
         << "property double y\n"
         << "property double z\n"
         << "property uchar red\n"
         << "property uchar green\n"
         << "property uchar blue\n"
         << "end_header\n";
    for(int i = 0; i < datas.size(); ++i) {
        if(datas[i].isInlier) {
            std::string pointStr = std::to_string(datas[i].pos[0]) + " " + \
                                   std::to_string(datas[i].pos[1]) + " " + \
                                   std::to_string(datas[i].pos[2]) + " " + \
                                   std::to_string(datas[i].color[2]) + " " + \
                                   std::to_string(datas[i].color[1]) + " " + \
                                   std::to_string(datas[i].color[0]) + "\n";
            file << pointStr;
        }
    }
    file.close();
}


void camera::test() {
    std::cout << "CAM ID " << id << std::endl;
}



