#pragma once
#include <vector>
struct Point{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[4];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};

std::vector<box> CreateAnchorRetinaFace(int w, int h);
void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);