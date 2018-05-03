//
// Created by chiliu on 5/2/18.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;
using namespace cv;


/****************************************************
 * This program shows how to use feature matching to
 * estimate extristric parameters (R T) of camera
 * **************************************************/
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& key_points_1,
                            std::vector<KeyPoint>& key_points_2,
                            std::vector< DMatch >& good_matches);

void pose_estimation_2d_2d( std::vector<KeyPoint> key_points_1,
                            std::vector<KeyPoint> key_points_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t);

Point2d pixel2cam ( const Point2d& p, const Mat& K);

int main(int argc, char** argv)
{
    if ( argc != 3)
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    std::vector<KeyPoint> key_points_1, key_points_2;
    vector<DMatch> good_matches;
    find_feature_matches (img_1, img_2, key_points_1, key_points_2,good_matches);

    Mat R,t;
    pose_estimation_2d_2d( key_points_1, key_points_2, good_matches, R, t );

    // validate E=t^R*scale
    Mat t_x = ( Mat_<double>(3,3)<<
                0,                  -t.at<double>( 2,0 ),    t.at<double>( 1,0 ),
                t.at<double>( 2,0 ), 0,                     -t.at<double>( 0,0 ),
               -t.at<double>( 1,0 ), t.at<double>(0,0),      0);
    cout<<"t^R"<<endl<<t_x*R<<endl;

    // validate epipolar constrain
    // camera internal reference fx fy u v, TUM Freiburg2
    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m: good_matches)
    {
        Point2d pt1 = pixel2cam( key_points_1[m.queryIdx].pt, K);
        Point2d pt2 = pixel2cam( key_points_2[m.trainIdx].pt, K);
        Mat y1 = ( Mat_<double>(3,1)<<pt1.x, pt1.y, 1);
        Mat y2 = ( Mat_<double>(3,1)<<pt2.x, pt2.y, 1);
        Mat d= y2.t()*t_x*R*y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& key_points_1,
                            std::vector<KeyPoint>& key_points_2,
                            std::vector< DMatch >& good_matches)
{
    // same as feature_extraction_chi
    // init
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
    // min_dist = min_element( matches.begin(), matches.end(), \
    // [](const DMatch& m1, const DMatch& m2) { return m1.distance < m2.distance;})->distance;
    // max_dist = max_element( matches.begin(), matches.end(), \
    // [](const DMatch& m1, const DMatch& m2) { return m1.distance < m2.distance;})->distance;

    printf ("-- Max distance : %f \n", max_dist);
    printf ("-- Min distance : %f \n", min_dist);

    // when distance > 2*min_distance, bad match delete
    // vector< DMatch > good_matches;
    for (int i = 0; i < matches[i].distance; i++ )
    {
        if ( matches[i].distance <= max ( 2 * min_dist, 50.0))
        {
            good_matches.push_back( matches[i]);
        }
    }
}


void pose_estimation_2d_2d( std::vector<KeyPoint> key_points_1,
                            std::vector<KeyPoint> key_points_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t)
{
    // camera internal reference fx fy u v, TUM Freiburg2
    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // convert point
    vector<Point2d> points_1;
    vector<Point2d> points_2;

    for ( int i = 0; i < (int)matches.size(); i++)
    {
        points_1.push_back ( key_points_1[matches[i].queryIdx].pt );
        points_2.push_back ( key_points_2[matches[i].trainIdx].pt );
    }

    // calculate foundamental martrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points_1, points_2, CV_FM_8POINT);
    cout<<"fundamental_matrix is"<<endl<< fundamental_matrix<<endl;

    //  calculate essential matrix
    Point2d principle_point ( 325.1, 249.7 );   // camera optical centre, TMU dataset
    double focal_length = 521;                  // camera focal length, TMU dataset
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points_1, points_2, focal_length, principle_point);
    cout<<"essential_matrix is"<<endl<< essential_matrix<<endl;

    // calculate homography matrix
    Mat homography_matrix;
    homography_matrix = findHomography (points_1, points_2, RANSAC, 3);
    cout<<"homography_matrix is"<<endl<< homography_matrix<<endl;

    // revocer R, T, form essential matrix
    recoverPose( essential_matrix, points_1, points_2, R, t, focal_length, principle_point);
    cout<<" R is "<<endl<<R<<endl;
    cout<<" t is "<<endl<<t<<endl;
}


Point2d pixel2cam ( const Point2d& p, const Mat& K)
{
    return Point2d
            (
                    (p.x - K.at<double>(0,2)) / K.at<double>(0,0),
                    (p.y - K.at<double>(1,2)) / K.at<double>(1,1)
            );
}