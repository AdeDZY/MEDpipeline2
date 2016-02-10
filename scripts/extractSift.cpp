#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> //Thanks to Alessandro

using namespace std;
int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread(argv[1], 0); //Load as grayscale

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::SiftFeatureDetector detector;
    cv::SiftDescriptorExtractor extractor;
    detector.detect(input, keypoints);
    extractor.compute(input, keypoints, descriptors);
    
    for(int i = 0; i < descriptors.rows; i++){
        for(int j = 0; j < descriptors.cols; j++){
            cout<<float(descriptors.at<float>(i, j))<<";";
        }
        cout<<endl;
    } 

    return 0;
}
