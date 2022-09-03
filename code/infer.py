import time
import os
import argparse
import shutil

import cv2
import numpy as np
import onnxruntime


def parse_arguments():
    parser = argparse.ArgumentParser(description='参数设置')
    # 训练参数
    parser.add_argument("--test_data_dir", default="/home/bing/图片/北京比赛测试集合", help="测试数据文件夹")
    parser.add_argument("--model_path", default="/home/bing/文档/说明文档/北京汽车喷漆检测比赛答辩项目/答辩文件/模型与推断脚本/seg.onnx", help="模型路径")
    parser.add_argument("--output_dir", default="./output_seg", help="模型输出路径")
    parser.add_argument("--threshold", default=0.35, help="模型输出路径")
    return parser.parse_args()


class Seg():
    def __init__(self, model_path, img_shape):
        self.session = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

        # 'TensorrtExecutionProvider' 'CUDAExecutionProvider' 'CPUExecutionProvider'
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = [img_shape[0], img_shape[1]]
        self.channel = img_shape[2]

    def predict(self, test_data_dir, threshold,output_dir):
        self.test_data_dir = test_data_dir
        self.threshold = threshold
        self.test_data_img_dir = os.path.join(self.test_data_dir, "image")
        self.test_data_true_label_dir = os.path.join(self.test_data_dir, "labels")

        self.predict_dir = os.path.join(output_dir, "predict")
        if os.path.exists(self.predict_dir):
            shutil.rmtree(self.predict_dir)
        os.makedirs(self.predict_dir)
        self.vis_dir = os.path.join(output_dir, "vis")
        if os.path.exists(self.vis_dir):
            shutil.rmtree(self.vis_dir)
        os.makedirs(self.vis_dir)


        for index,img_file_name in enumerate(os.listdir(self.test_data_img_dir)):
            img_file_path = os.path.join(self.test_data_img_dir, img_file_name)
            img = cv2.imdecode(np.fromfile(img_file_path, dtype=np.uint8), 1)  # 可读取中文路径图片
            true_label_path = os.path.join(self.test_data_true_label_dir, img_file_name)
            true_label = cv2.imdecode(np.fromfile(true_label_path, dtype=np.uint8), 1)  # 可读取中文路径图片

            result_img, predict_result = self.segment_predict_one(img, true_label, threshold=self.threshold)

            predict_result_path = os.path.join(self.predict_dir,img_file_name)
            cv2.imwrite(predict_result_path,predict_result)

            vis_result_path = os.path.join(self.vis_dir, img_file_name)
            cv2.imwrite(vis_result_path, result_img)
            # cv2.imshow("result_img",result_img)
            # cv2.imshow("predict_result", predict_result)
            # cv2.imwrite("result_img.png",result_img)
            # cv2.imwrite("predict_result.png", predict_result)
            # cv2.waitKey(0)

            print("infer {0} images ".format(index))

        pass

    def segment_predict_one(self, image, true_label, threshold=0.5):
        image = (np.array(image[:, :]))
        image_data = cv2.dnn.blobFromImage(image,
                                           scalefactor=1 / 255.0,
                                           swapRB=False,
                                           crop=False,
                                           size=(self.input_size[1], self.input_size[0]))
        im = image_data

        out = self.session.run([], {self.input_name: im})[0]

        ##################
        result_img = out[0, :, :, :]
        result = result_img

        result = (result * 255.0).astype(np.uint8)[0]
        result = np.clip(result, 0, 255)

        b = np.zeros((result.shape[0], result.shape[1]), dtype=result.dtype)
        g = np.zeros((result.shape[0], result.shape[1]), dtype=result.dtype)
        r = np.zeros((result.shape[0], result.shape[1]), dtype=result.dtype)

        result_rgb = np.dstack([result, result, r])

        result_merge_rgb = cv2.addWeighted(image, 0.5, result_rgb, 0.5, 0)

        retval, im_at_fixed = cv2.threshold(result, int(255 * threshold), 255, cv2.THRESH_BINARY)

        result_binary = np.dstack([im_at_fixed, im_at_fixed, r])
        result_merge_binary = cv2.addWeighted(image, 0.5, result_binary, 0.5, 0)

        true_binary = true_label
        result_merge_true_binary = cv2.addWeighted(image, 0.5, true_binary, 0.5, 0)

        result_img1 = np.hstack((result_merge_rgb, image, result_rgb))
        result_img2 = np.hstack((result_merge_binary, image, result_binary))
        result_img3 = np.hstack((result_merge_true_binary, image, true_binary))

        result_img = np.vstack((result_img1, result_img2, result_img3))
        return result_img, result

    def test_speed(self):
        print(" start test speed! ")
        img_list = os.listdir(self.test_data_img_dir)
        img_path = os.path.join(self.test_data_img_dir,img_list[0])
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        image = (np.array(image[:, :]))

        image_data = cv2.dnn.blobFromImage(image,
                                           scalefactor=1 / 255.0,
                                           swapRB=False,
                                           crop=False,
                                           size=(self.input_size[1], self.input_size[0]))
        im = image_data

        for i in range(2):
            out = self.session.run([], {self.input_name: im})[0]
        t0 = time.time()
        for i in range(50):
            out = self.session.run([], {self.input_name: im})[0]
        speed = (time.time() - t0)/50
        print("seg 1 img cost {} seconds!".format(speed))
        return speed


class CountResult():
    def __init__(self, true_label_dir, predict_label_dir):
        self.true_label_path_list = sorted(
            [os.path.join(true_label_dir, true_label_file) for true_label_file in os.listdir(true_label_dir)])
        self.predict_label_path_list = sorted(
            [os.path.join(predict_label_dir, predict_label_file) for predict_label_file in
             os.listdir(predict_label_dir)])

    def count_nums(self, threshold=0.5):
        print("start count!")

        iou_list = []
        recall_list = []
        precision_list = []
        TP_list = []
        FP_list = []
        FN_list = []
        TN_list = []

        for index, label_img_path in enumerate(self.true_label_path_list):
            predict_img_gray_path = self.predict_label_path_list[index]

            label_img_gray = cv2.imread(label_img_path, 0)
            retval, label_img = cv2.threshold(label_img_gray, int(255 * threshold), 255, cv2.THRESH_BINARY)

            predict_img_gray = cv2.imread(predict_img_gray_path, 0)
            retval, predict_img = cv2.threshold(predict_img_gray, int(255 * threshold), 255, cv2.THRESH_BINARY)

            label_img_reshape = label_img.reshape(-1)
            predict_img_reshape = predict_img.reshape(-1)

            label_num = np.bincount(label_img_reshape, weights=None, minlength=None)
            label_back_num = label_num[0]
            if len(label_num) == 1:
                label_front_num = 0
            else:
                label_front_num = label_num[-1]

            predict_num = np.bincount(predict_img_reshape, weights=None, minlength=None)
            predict_back_num = predict_num[0]
            if len(predict_num) == 1:
                predict_front_num = 0
            else:
                predict_front_num = label_num[-1]

            temp = np.bincount((label_img_reshape * predict_img_reshape), weights=None, minlength=None)
            if len(temp) == 1:
                label_front_predict_front = 0
            else:
                label_front_predict_front = temp[-1]

            label_front_predict_back = label_front_num - label_front_predict_front

            label_back_predict_front = predict_front_num - label_front_predict_front

            label_back_predict_back = label_back_num - label_back_predict_front

            TP = label_front_predict_front
            FP = label_front_predict_back
            FN = label_back_predict_front
            TN = label_back_predict_back

            if TP == 0:
                TP = 0.00000001

            recall = TP / (TP + FN)
            precision = TP / (TP + FP)

            iou = TP / (TP + FP + FN)

            iou_list.append(iou)
            recall_list.append(recall)
            precision_list.append(precision)

            # cv2.imshow("label_img", label_img)
            # cv2.imshow("predict_img_gray", predict_img_gray)
            # cv2.imshow("predict_img", predict_img)

            # cv2.waitKey(0)
            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            TN_list.append(TN)

            # print(index)

        iou_result = sum(iou_list) / len(iou_list)
        recall_result = sum(recall_list) / len(recall_list)
        precision_result = sum(precision_list) / len(precision_list)

        TTP = sum(TP_list)
        FFP = sum(FP_list)
        FFN = sum(FN_list)
        TTN = sum(TN_list)

        bb = 4
        return iou_result,recall_result,precision_result


if __name__ == "__main__":
    args = parse_arguments()
    img_shape = [480, 960, 3]
    model_path = args.model_path
    test_data_dir = args.test_data_dir
    threshold = float(args.threshold)
    output_dir=args.output_dir
    seg = Seg(model_path=model_path, img_shape=img_shape)
    seg.predict(test_data_dir=test_data_dir,threshold=threshold,output_dir=output_dir)
    speed = seg.test_speed()

    true_label_dir = seg.test_data_true_label_dir
    predict_label_dir = seg.predict_dir

    cr = CountResult(true_label_dir, predict_label_dir)
    iou_result,recall_result,precision_result = cr.count_nums(threshold=threshold)
    print("iou_result: {0}".format(iou_result))
    print("recall_result: {0}".format(recall_result))
    print("precision_result: {0}".format(precision_result))

