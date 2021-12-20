#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("/src/image", 1);

  std::string src_path;
  nh.getParam("/publisher/src_path", src_path);

  sensor_msgs::ImagePtr msg;
  // cv::Mat image = cv::imread(src_path, cv::IMREAD_COLOR);
  // msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  cv::VideoCapture capture;
  capture.open(src_path);
  if (!capture.isOpened()){
    std::cout << "Could not load video..." << std::endl;
    return -1;
  }

  cv::Mat frame;
  ros::Rate loop_rate(5);
  while (nh.ok()) {
    capture >> frame;
    if (!frame.empty()) msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}

