
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

bool DEBUG_CAMFUSION = false;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    double angle = 0;

    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((topviewImg.cols-1)/2.0, (topviewImg.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), topviewImg.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - topviewImg.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - topviewImg.rows/2.0;

    // ratate and resize image
    cv::Mat dst;
    cv::warpAffine(topviewImg, dst, rot, bbox.size());
    cv::resize(dst, dst, cv::Size(dst.cols/3, dst.rows/3));
    
    // display image
    // string windowName = "3D Objects";
    // cv::namedWindow(windowName, 1);
    // cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        // display image
        string windowName = "3D Objects";
        cv::namedWindow(windowName, 6);
        cv::imshow(windowName, dst);
        cout << "Press key to continue to next frame" << endl;
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // todo : find all keypoint matches that belong to each 3D object
    std::vector<cv::DMatch> tempMatches;
    for( auto kptMatch : kptMatches ){
        // cv::KeyPoint pPrev = kptsPrev[kptMatch.queryIdx];
        cv::KeyPoint pCurr = kptsCurr[kptMatch.trainIdx];
        if( boundingBox.roi.contains(pCurr.pt) )
            tempMatches.push_back(kptMatch);
    }
    vector<double> distances; 
    for( auto itr = tempMatches.begin(); itr != tempMatches.end(); itr++ ){
        cv::KeyPoint pPrev = kptsPrev[itr->queryIdx];
        cv::KeyPoint pCurr = kptsCurr[itr->trainIdx];
        distances.push_back(sqrt(pow(pPrev.pt.x - pCurr.pt.x, 2) + pow(pPrev.pt.y - pCurr.pt.y, 2)));
    }
    
    double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
    double mean = sum / distances.size();

    std::vector<double> diff(distances.size());
    std::transform(distances.begin(),distances.end(),diff.begin(),std::bind2nd(std::minus<double>(),mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / distances.size());

    for( auto i =0; i < tempMatches.size(); i++ )
        if( abs(distances[i] - mean) < stdev )
            boundingBox.kptMatches.push_back(kptMatches[i]);
    if(DEBUG_CAMFUSION)
        std::cout << "Filtered kptMatches number : " << boundingBox.kptMatches.size() << std::endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1){ // outer kpt. loop
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop
            double minDist = 100.0; // min. required distance
            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0){
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    // compute median dist. ratio to remove outlier influence
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; 

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC){

    vector<double> dPrev, dCurr;
    int prevSize = lidarPointsPrev.size();
    int currSize = lidarPointsCurr.size();
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) {
        dPrev.push_back(it->x);
    }
    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) {
        dCurr.push_back(it->x);
    }
    double dT = 1 / frameRate;

    std::sort(dPrev.begin(), dPrev.end());
    std::sort(dCurr.begin(), dCurr.end());

    long medIndexPrev = floor(prevSize / 2.0);
    double xPrev = prevSize % 2 == 0 ? (dPrev[medIndexPrev - 1] + dPrev[medIndexPrev]) / 2.0 : dPrev[medIndexPrev]; 

    long medIndexCurr = floor(currSize / 2.0);
    double xCurr = currSize % 2 == 0 ? (dCurr[medIndexCurr - 1] + dCurr[medIndexCurr]) / 2.0 : dCurr[medIndexCurr]; 
    
    TTC = xCurr * dT / (xPrev - xCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // cv::DMatch reference : https://docs.opencv.org/3.4/d4/de0/classcv_1_1DMatch.html  

    int prevBoxNum = prevFrame.boundingBoxes.size();
    int currBoxNum = currFrame.boundingBoxes.size();
	int p_count[prevBoxNum][currBoxNum] = {};
    for (auto match : matches) {
    	cv::KeyPoint query = prevFrame.keypoints[match.queryIdx];
    	cv::KeyPoint train = currFrame.keypoints[match.trainIdx];

        bool is_query_found = false;
        bool is_train_found = false;

    	int query_id = 0;
    	int train_id = 0;

        for( int i = 0; i < prevBoxNum; i++ ){
            if ( prevFrame.boundingBoxes[i].roi.contains( query.pt ) ){
                is_query_found = true;
                query_id = i;
                break;
            }
        }
        for( int i = 0; i < currBoxNum; i++ ){
            if ( currFrame.boundingBoxes[i].roi.contains( train.pt ) ){
                is_train_found = true;
                train_id = i;
                break;
            }
        }
    	if (is_query_found && is_train_found) {
            p_count[query_id][train_id] += 1;
    	}
    }
    for (int i = 0; i < prevBoxNum; i++){
        int id, max = 0;
        for (int j = 0; j < currBoxNum; j++){
            if( max < p_count[i][j] ){
                max = p_count[i][j];
                id = j;
            }
        }
        bbBestMatches[i] = id;
    }

    if (DEBUG_CAMFUSION)
        for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
             std::cout << "Box " << i << " matched to " << bbBestMatches[i] << std::endl;
}
