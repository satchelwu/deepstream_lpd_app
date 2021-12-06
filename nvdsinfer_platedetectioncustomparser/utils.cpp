#include "utils.h"
#include <cmath>

std::vector<box> CreateAnchorRetinaFace(int w, int h)
{
    std::vector<box> anchor;
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(std::ceil(h/steps[i]));
        feature_map[i].push_back(std::ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {24, 48};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {96, 192};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {192, 384};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                    //std::cout<< cx << " "<< cy << " "<<s_kx << " "<<s_ky << " "<< std::endl;
                }
            }
        }

    }
    return anchor;
}

void nms(std::vector<bbox> &input_boxes, float NMS_THRESH) 
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter + 1E-6);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}