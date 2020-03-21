# SFND 3D Object Tracking

### Project Status:

![issue_badge](https://img.shields.io/badge/build-Passing-green) ![issue_badge](https://img.shields.io/badge/UdacityRubric-Passing-green)

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## FP.1 : Match 3D Objects
> implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property)“. Matches must be the ones with the highest number of keypoint correspondences.

```c++
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
```

## FP.2 : Compute Lidar-based TTC
> compute the time-to-collision for all matched 3D objects based on Lidar measurements alone

```c++
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
```

## FP.3 : Associate Keypoint Correspondences with Bounding Boxes
> find all keypoint matches that belong to each 3D object.

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // todo : find all keypoint matches that belong to each 3D object
    float distanceSum = 0.0f;
    int inlierNum = 0;
    for(cv::DMatch& kptMatch : kptMatches) {
        if(boundingBox.roi.contains(kptsCurr[kptMatch.trainIdx].pt)){
            distanceSum += kptMatch.distance; 
            inlierNum++;
        }
    }
    if(inlierNum == 0)
        return;
    float avgDistance = distanceSum /= inlierNum;
    for(cv::DMatch& kptMatch : kptMatches) {
        cv::Point pCurr = kptsCurr[kptMatch.trainIdx].pt;
        if(boundingBox.roi.contains(pCurr) && kptMatch.distance < avgDistance) {
            boundingBox.kptMatches.push_back(kptMatch);
        }
    }

    if(DEBUG_CAMFUSION)
        std::cout << "Filtered kptMatches number : " << boundingBox.kptMatches.size() << std::endl;
}
```

## FP.4 : Compute Camera-based TTC
```c++
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
```

## FP.5 : Performance Evaluation 1
> Look for several examples where you have the impression that the Lidar-based TTC estimate is way off. Once you have found those, describe your observations and provide a sound argumentation why you think this happened.

If you see the consencutive pictures by manually, you can catch that front vehicle is getting closer to camera (also closer to Lidar.)
But in the step between picture 2~5, camera TTC goes up, moreover Lidar TTC suddenly goes up to 20seconds.
Maybe the reason for that is...
1. Sensor erroneous (ex - light reflection)
2. Velocity of front vehicle goes up actually (since this TTC model is based on Constant velocity model)
3. TTC is calculated based on outlier ( This can be happend, since I used median when choosing the point. Moreover the std-variant at picture 3 seems like larger than other pictures )

## FP.6 : Performance Evaluation 2
> running the different detector / descriptor combinations and looking at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off.

Which detector /descriptor combination is best -> In Mid-term project, I found that 
1. FAST + BRIEF
2. ORB + BRIEF
3. BRIEF + BRISK
those combinations are suitable for our case.

Let's see whether those things still best in this case
Here's the [Google SpreadSheet](https://docs.google.com/spreadsheets/d/175UXWGtkNnRr4e7rKXONoMUjOBZpzO9gMPYLuDNd5Pw/edit?usp=sharing) for comparing.

![Screenshot from 2020-02-27 14-15-07](https://user-images.githubusercontent.com/12381733/75414359-a4d4be80-596b-11ea-913a-9191d024dc92.png)

According to this report, AZAKE shows great result for calculating TTC.
Moreover FAST + BRIEF / BRIEF + BRISK combination also performed pretty good results.
But, There was some outliers in ORB results. 
