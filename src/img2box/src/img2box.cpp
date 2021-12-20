#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <dnn_detect/DetectedObjectArray.h>
//#include <iostream>

static void sleep_ms(unsigned int secs)
{
    struct timeval tval;
    tval.tv_sec=secs/1000;
    tval.tv_usec=(secs*1000)%1000000;
    select(0,NULL,NULL,NULL,&tval);
}


class ImageConverter
{
private:
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher box_pub_;
    torch::jit::script::Module model_;
    const int kIMAGE_H_ = 640;
    const int kIMAGE_W_ = 640;
    float score_thresh_;
    float iou_thresh_;
    torch::DeviceType device_;
    std::vector<std::string> classnames_;
    std::string pt_path;
    std::string class_path;
    bool use_gpu;
    std::time_t start;
    bool trigger;
    float delta;

public:
    ImageConverter(ros::NodeHandle nh):
    it_(nh), start(std::time(0)), trigger(true)
    {
        nh.getParam("/detector/score_thresh", score_thresh_);
        nh.getParam("/detector/iou_thresh", iou_thresh_);
        nh.getParam("/detector/pt_path", pt_path);
        nh.getParam("/detector/class_path", class_path);
        nh.getParam("/detector/use_gpu", use_gpu);
        nh.getParam("/detector/delta", delta);

        if (use_gpu)
            device_ = torch::kCUDA;
        else
            device_ = torch::kCPU;
     	image_sub_ = it_.subscribe("/src/image", 1, &ImageConverter::inference, this);  //TODO "/mynteye/left_rect/image_rect"
        image_pub_ = it_.advertise("/prediction/image", 1);
      	box_pub_ = nh.advertise<dnn_detect::DetectedObjectArray>("/prediction/box", 1);
    	model_ = torch::jit::load(pt_path);
    	model_.to(device_);
        std::ifstream f(class_path);
        std::string name = "";
        while (std::getline(f, name))
        {
            classnames_.push_back(name);
        }
	//cv::namedWindow(OPENCV_WINDOW);
    }

    //~ImageConverter()
    //{
    //    cv::destroyWindow(OPENCV_WINDOW);
    //}

    void inference(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");  // convert ros msgs to opencv format
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        //preprocess
	    torch::Tensor in_tensor = ImageConverter::preprocess(cv_ptr->image);

        //predict
        torch::Tensor preds = model_.forward({in_tensor}).toTuple()->elements()[0].toTensor();
        preds = preds.to(torch::kCPU);

        //postprocess
	    ImageConverter::postprocess(cv_ptr->image, preds);

    }

    std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5)
    {
        std::vector<torch::Tensor> output;
        for (size_t i=0; i < preds.sizes()[0]; ++i)
        {
            torch::Tensor pred = preds.select(0, i);

            // Filter by scores
            torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
            pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
            if (pred.sizes()[0] == 0) continue;

            // (center_x, center_y, w, h) to (left, top, right, bottom)
            pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
            pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
            pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
            pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

            // Computing scores and classes
            std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
            pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
            pred.select(1, 5) = std::get<1>(max_tuple);

            torch::Tensor  dets = pred.slice(1, 0, 6);

            torch::Tensor keep = torch::empty({dets.sizes()[0]});
            torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
            std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
            torch::Tensor v = std::get<0>(indexes_tuple);
            torch::Tensor indexes = std::get<1>(indexes_tuple);
            int count = 0;
            while (indexes.sizes()[0] > 0)
            {
                keep[count] = (indexes[0].item().toInt());
                count += 1;

                // Computing overlaps
                torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
                for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
                {
                    lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                    tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                    rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                    bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                    widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                    heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
                }
                torch::Tensor overlaps = widths * heights;

                // FIlter by IOUs
                torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
                indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
            }
            keep = keep.toType(torch::kInt64);
            output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
        }
        return output;
    }

    torch::Tensor preprocess(const cv::Mat &frame)
    {
            if (frame.empty() || !frame.data)
            {
                std::cout << "Frame empty!" << std::endl;
            exit(-1);
            }

            // resize and to Tensor
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(kIMAGE_W_, kIMAGE_H_));
            torch::Tensor in_tensor = torch::from_blob(resized_frame.data, {kIMAGE_H_, kIMAGE_W_, 3}, torch::kByte);
            in_tensor = in_tensor.unsqueeze(0).permute({0, 3, 1, 2}).toType(torch::kFloat).div(255); //TODO to(DEVICE)
            in_tensor = in_tensor.to(device_);
            return in_tensor;
    }

    void postprocess(cv::Mat &frame, torch::Tensor &preds)
    {
        dnn_detect::DetectedObjectArray box_msgs;
        std::vector<torch::Tensor> dets = non_max_suppression(preds, score_thresh_, iou_thresh_);
        std::time_t interval;

        if (trigger) start = std::time(0);
        if (dets.size() > 0){
            trigger = false;
            interval = std::time(0) - start;
        }else{
            trigger = true;
        }
        cv::Scalar color;
        if (interval >= delta){
            color = cv::Scalar(0, 0, 255);
        }else{
            color = cv::Scalar(255, 0, 0);
        }
        cv::putText(frame,
            "Tracking Time(s): " + cv::format("%d", int(interval)),
            cv::Point(30, 30),
			cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);

        if (dets.size() > 0)
        {
            // Visualize result
            for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
            {
                float left = dets[0][i][0].item().toFloat() * frame.cols / kIMAGE_W_;
                float top = dets[0][i][1].item().toFloat() * frame.rows / kIMAGE_H_;
                float right = dets[0][i][2].item().toFloat() * frame.cols / kIMAGE_W_;
                float bottom = dets[0][i][3].item().toFloat() * frame.rows / kIMAGE_H_;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();

                dnn_detect::DetectedObject obj;
                obj.confidence = score;
                obj.class_name = classnames_[classID];
                obj.x_min = top;
                obj.x_max = bottom;
                obj.y_min = left;
                obj.y_max = right;
                box_msgs.objects.push_back(obj);

                cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), color, 2);
				cv::putText(frame,
					classnames_[classID] + ": " + cv::format("%.2f", score),
					cv::Point(left, top),
					cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, color, 2);
            }
        }
        if (interval >= delta) box_pub_.publish(box_msgs);
        sensor_msgs::ImagePtr image_msgs = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();  //convert opencv format to ros msgs
        image_pub_.publish(image_msgs);
        return;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "detector");
    ros::NodeHandle nh;
    ImageConverter ic(nh);
    ros::spin();
    return 0;
}

