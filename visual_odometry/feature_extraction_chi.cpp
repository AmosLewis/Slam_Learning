//
// Created by chiliu on 5/1/18.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/core/version.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{

    if (CV_MAJOR_VERSION == 2)
    {
        cout<<"opencv2";
    } else if (CV_MAJOR_VERSION == 3)
    {
        cout<<"opencv3"<<endl;
    }

    if ( argc != 3)
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
    cout<<"input img1 complete "<<endl;
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    cout<<"input img2 complete "<<endl;

    // init
    std::vector<KeyPoint> key_points_1, key_points_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );

    // detect Oriented Fast key points by image
    detector->detect( img_1, key_points_1 );
    detector->detect( img_2, key_points_2 );

    // calculate descriptor by key points
    descriptor->compute( img_1, key_points_1, descriptors_1 );
    descriptor->compute( img_2, key_points_2, descriptors_2 );

    // show key_points and save image
    Mat out_image1;
    drawKeypoints(img_1, key_points_1, out_image1);
    imshow("ORB descriptor", out_image1);
    imwrite("ORB_descriptor.jpg", out_image1);

    // match BRIEF descriptors of two image
    vector<DMatch> matches;
    matcher->match ( descriptors_1, descriptors_2, matches );

    // filter good matches
    double min_dist = 10000, max_dist = 0;
    // find min_dist, max_dist
    for ( int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if ( dist < min_dist) min_dist = dist;
        if ( dist > max_dist) max_dist = dist;
    }

    // another way for find min_dist, max_dist
    min_dist = min_element( matches.begin(), matches.end(), \
    [](const DMatch& m1, const DMatch& m2) { return m1.distance < m2.distance;})->distance;
    max_dist = max_element( matches.begin(), matches.end(), \
    [](const DMatch& m1, const DMatch& m2) { return m1.distance < m2.distance;})->distance;

    printf ("-- Max distance : %f \n", max_dist);
    printf ("-- Min distance : %f \n", min_dist);

    // when distance > 2*min_distance, bad match delete
    vector< DMatch > good_matches;
    for (int i = 0; i < matches[i].distance; i++ )
    {
        if ( matches[i].distance <= max ( 2 * min_dist, 40.0))
        {
            good_matches.push_back( matches[i]);
        }
    }

    // draw result
    Mat image_matches;
    Mat image_good_matches;
    drawMatches(img_1, key_points_1, img_2, key_points_2, matches, image_matches);
    drawMatches(img_1, key_points_1, img_2, key_points_2, good_matches, image_good_matches);
    imshow( "ALL matches", image_matches);
    imwrite( "ALl_matches.jpg", image_matches);
    imshow( "ALL good matches", image_good_matches);
    imwrite( "ALl_good_matches.jpg", image_good_matches);
    printf("-- num of good_matches : %d ", good_matches.size());

    waitKey(0);

    return 0;
}

