/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cstring>
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include "utils.h"

inline bool cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

extern "C" bool NvDsInferParseCustomPlateDetection(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                                   NvDsInferNetworkInfo const &networkInfo,
                                                   NvDsInferParseDetectionParams const &detectionParams,
                                                   std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    auto layer_finder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *
    {
        for (auto &layer : outputLayersInfo)
        {
            if (layer.dataType == FLOAT &&
                (layer.layerName && name == layer.layerName))
            {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *box_layer = layer_finder("output_0");
    const NvDsInferLayerInfo *score_layer = layer_finder("output_1");
    const NvDsInferLayerInfo *landmark_layer = layer_finder("output_2");
    
    if (!score_layer || !landmark_layer || !box_layer)
    {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }
    int net_width = networkInfo.width;
    int net_height = networkInfo.height;
    static std::vector<box> anchors = CreateAnchorRetinaFace(net_width, net_height);
    std::vector<bbox> total_box;
    float *box_data_ptr = (float*)box_layer->buffer;
    float *score_data_ptr = (float*)score_layer->buffer;
    float *landmark_data_ptr = (float*)landmark_layer->buffer;
    float nms_threshold = 0.25;
    float score_threshold = 0.25;
    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchors.size(); ++i)
    {
        if (*(score_data_ptr+1) > score_threshold)
        {
            box anchor_box = anchors[i];
            box regresion_box;
            bbox result;
            // loc and conf
            regresion_box.cx = anchor_box.cx + *box_data_ptr * 0.1 * anchor_box.sx;
            regresion_box.cy = anchor_box.cy + *(box_data_ptr+1) * 0.1 * anchor_box.sy;
            regresion_box.sx = anchor_box.sx * exp(*(box_data_ptr+2) * 0.2);
            regresion_box.sy = anchor_box.sy * exp(*(box_data_ptr+3) * 0.2);

            result.x1 = (regresion_box.cx - regresion_box.sx/2) * net_width;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (regresion_box.cy - regresion_box.sy/2) * net_height;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (regresion_box.cx + regresion_box.sx/2) * net_width ;
            if (result.x2>net_width - 1)
                result.x2 = net_width -1;
            result.y2 = (regresion_box.cy + regresion_box.sy/2)* net_height;
            if (result.y2>net_height - 1)
                result.y2 = net_height - 1;
            result.s = *(score_data_ptr + 1);

            // landmark
            for (int j = 0; j < 4; ++j)
            {
                result.point[j]._x =( anchor_box.cx + *(landmark_data_ptr + (j<<1)) * 0.1 * anchor_box.sx ) * net_width;
                result.point[j]._y =( anchor_box.cy + *(landmark_data_ptr + (j<<1) + 1) * 0.1 * anchor_box.sy ) * net_height;
            }

            total_box.push_back(result);
        }
        box_data_ptr += 4;
        score_data_ptr += 2;
        landmark_data_ptr += 8;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, nms_threshold);
    for(int i = 0; i < total_box.size(); ++i)
    {
        bbox& bbox = total_box[i];
        NvDsInferObjectDetectionInfo obj;
        obj.classId = 0;
        obj.detectionConfidence = bbox.s; 
        obj.left = bbox.x1;
        obj.top = bbox.y1;
        obj.width = bbox.x2 - bbox.x1 + 1;
        obj.height = bbox.y2 - bbox.y1 + 1;
        // obj.landmarks.resize(8);
        // memcpy(obj.landmarks.data(), bbox.point, sizeof(Point) * 4);
        std::cout << bbox.s << ", " <<  obj.left <<  ", " << obj.top << ", " << obj.width << ", " << obj.height << std::endl;
        if(obj.width && obj.height)
        {
            objectList.emplace_back(obj);
        } 
    }
    return true;
}
/* Check that the custom function has been defined correctly */

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPlateDetection);
