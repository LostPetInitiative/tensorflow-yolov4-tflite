import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import os
import multiprocessing
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from concurrent.futures import ThreadPoolExecutor
import asyncio


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416-coco',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
#flags.DEFINE_string('output', 'result.png', 'path to output image')
#flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

cpuCount = multiprocessing.cpu_count()

# cat class 15
# dog class 16

def main(_argv):
    async def work():
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size

        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        print("Loaded model {0}".format(FLAGS.weights))
        infer = saved_model_loaded.signatures['serving_default']

        threadPool = ThreadPoolExecutor(max_workers=cpuCount*2)    
        gpuPool = ThreadPoolExecutor(max_workers=1)    
        loop = asyncio.get_running_loop()
        
        def loadImage(imagePath):
            original_image = cv2.imread(imagePath)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.
            # image_data = image_data[np.newaxis, ...].astype(np.float32)
            return original_image, image_data

        def rotateImage(image_data, rotation):
            images_data = []
            images_data.append(np.rot90(image_data, k=rotation))
            images_data = np.asarray(images_data).astype(np.float32)
            return images_data, rotation
        
        def getBboxes(images_data, targetClass:int):
            #print("images_data shape {0}".format(images_data.shape))
        
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data) # works only for batch size 1 and 2 for some reason
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    # A 4-D float Tensor of shape [batch_size, num_boxes, q, 4].
                    # If q is 1 then same boxes are used for all classes otherwise,
                    # if q is equal to number of classes, class-specific boxes are used. 
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    # A 3-D float Tensor of shape [batch_size, num_boxes, num_classes]
                    # representing a single score corresponding to each box (each row of boxes). 
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])), \
                    # the maximum number of boxes to be selected by non-max suppression per class 
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.5,
                score_threshold=FLAGS.score
            )
            # 'nmsed_boxes':    A [batch_size, max_detections, 4] float32 tensor containing
            #                   the non-max suppressed boxes.
            # 'nmsed_scores':   A [batch_size, max_detections] float32 tensor containing
            #                   the scores for the boxes.
            #  'nmsed_classes': A [batch_size, max_detections] float32 tensor containing
            #                   the class for boxes.
            #  'valid_detections':  A [batch_size] int32 tensor indicating
            #                       the number of valid detections per batch item.
            #                       Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
            #                       The rest of the entries are zero paddings.
            
            # here we leave only target class

            # boxes = tf.squeeze(boxes, axis=0)
            # scores = tf.squeeze(scores, axis=0)
            classesSqueezed = tf.squeeze(classes, axis=0)

            validClassMask = tf.equal(classesSqueezed,targetClass)
            validClassIndices = tf.squeeze(tf.where(validClassMask),axis=1)
            validClassCount = tf.shape(validClassIndices)

            boxes = tf.gather(boxes, validClassIndices, axis=1)
            scores = tf.gather(scores, validClassIndices, axis=1)
            classes = tf.gather(classes, validClassIndices, axis=1)

            #print("valid class indices {0}".format(validClassIndices))

            npBoxes = boxes.numpy()
            npScores = scores.numpy()
            npClasses = classes.numpy()
            npValidDetections = validClassCount.numpy()

            pred_bbox = [npBoxes, npScores, npClasses, npValidDetections]
            return pred_bbox

        async def FindBestRotation(imagePath, targetClass):
            """Returns (annotatedImage,extractedPetImage, bestScore, bestRotation) or quadro-tuple on Nones if the pet is not detected"""
            original_image, image_data = await loop.run_in_executor(threadPool, loadImage, imagePath)
            loadImageJobs = [loop.run_in_executor(threadPool,rotateImage, image_data, k) for k in range(4)]
            bestScore = 0.0
            bestScoreBboxes = None
            bestRotation = 0
            for coro in asyncio.as_completed(loadImageJobs):
                images_data, rotation = await coro
                bboxes = await loop.run_in_executor(gpuPool,getBboxes, images_data, targetClass)
                score = bboxes[1]
                scoreShape = score.shape
                detectedCount = scoreShape[1]
                if detectedCount != 1:
                    continue # we are interested in the case when only single pet detected
                scoreVal = score[0,0]
                if scoreVal > bestScore: # selecting the rotation which gives the highest score
                    bestScore = scoreVal
                    bestScoreBboxes = bboxes
                    bestRotation = rotation
            if bestScoreBboxes != None:
                # did not find any rotation with single pet present
                if bestRotation != 0:
                    rotatedImage = np.rot90(original_image,k=bestRotation)
                else:
                    rotatedImage = original_image

                annotatedImage = np.copy(rotatedImage)
                annotatedImage = utils.draw_bbox(annotatedImage, bestScoreBboxes)
                annotatedImage = Image.fromarray(annotatedImage.astype(np.uint8))
                annotatedImage = cv2.cvtColor(np.array(annotatedImage), cv2.COLOR_BGR2RGB)


                #image_h, image_w, _ = rotatedImage.shape
                coor = bestScoreBboxes[0][0,0,:].astype(np.int32) # y1,x1,y2,x2
                print("bbox {0}".format(coor))
                # coor[0] = int(coor[0] * image_h)
                # coor[2] = int(coor[2] * image_h)
                # coor[1] = int(coor[1] * image_w)
                # coor[3] = int(coor[3] * image_w)

                print("rotated shape {0}".format(rotatedImage.shape))
                extracted = rotatedImage[coor[0]:coor[2], coor[1]:coor[3], :]
                extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
                return annotatedImage, extracted, bestScore, bestRotation
            else:
                return None, None, None, None
        
        annotatedImage,extractedImage, bestScore,rotation = await FindBestRotation("/mnt/ML/LostPetInitiative/pet911ru/rf240266/424288.jpg", 16)

        cv2.imwrite(os.path.join("/mnt/ssd/PetSimilarity-ML/test","annotated.jpg"), annotatedImage)
        cv2.imwrite(os.path.join("/mnt/ssd/PetSimilarity-ML/test","extracted.jpg"), extractedImage)
        print("best score {0} at rotation {1}".format(bestScore, rotation))
    asyncio.run(work())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
