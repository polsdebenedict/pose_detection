import argparse
import cv2
import os
import pickle
from shutil import copyfile

from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose pose extraction method')

    parser.add_argument("--method", default='detectron', help='choose detectron, vibe')
    parser.add_argument("--input_video_folder", default='./output/video_sample/mask/',
                        help='specify folder containing video to process')
    parser.add_argument("--output_pose_folder", default='./output/video_sample/mask',
                        help='specify folder to save joint output')
    parser.add_argument("--get", default='keypoints',
                        help='specify what you want to get with detectron: keypoints or mask')
    parser.add_argument("--annot_input", default='-',
                        help='specify specify if you want to add the annotation file to the results')

    args = vars(parser.parse_args())

    videos = os.listdir(args['input_video_folder'])

    if args["method"] == "detectron":

        if not os.path.isdir(args['output_pose_folder'] + "detectron/"):
            os.mkdir(args['output_pose_folder'] + "detectron/")

        for el in videos:

            file_name = el[:-4]

            if not os.path.isdir(args['output_pose_folder'] + "detectron/" + file_name):

                os.mkdir("./output/joint/detectron/" + file_name)
                path = './output/video/' + el
                video = cv2.VideoCapture(path)

                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = video.get(cv2.CAP_PROP_FPS)
                num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                
                # Initialize predictor
                cfg = get_cfg()  # get a fresh new config

                if args['get'] == 'keypoints':
                    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

                elif args['get'] == 'mask':
                    cfg.merge_from_file(
                        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

                predictor = DefaultPredictor(cfg)
                # Initialize visualizer
                v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

                tot_out = []

                def runOnVideo(video, maxFrames):
                    """ Runs the predictor on every frame in the video (unless maxFrames is given),
                    and returns the frame with the predictions drawn.
                    """

                    readFrames = 0
                    while True:
                        hasFrame, frame = video.read()
                        if not hasFrame:
                            break

                        # Get prediction results for this frame
                        outputs = predictor(frame)

                        # Make sure the frame is colored
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Draw a visualization of the predictions using the video visualizer
                        visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

                        # Convert Matplotlib RGB format to OpenCV BGR format
                        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

                        yield visualization, outputs

                        readFrames += 1
                        if readFrames > maxFrames:
                            break


                # Create a cut-off for debugging
                num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                # Enumerate the frames of the video
                for visualization, outputs in tqdm(runOnVideo(video, num_frames), total=num_frames):
                    # Write test image
                    # cv2.imwrite('POSE detectron2.png', visualization)

                    # Write to video file
                    video_writer.write(visualization)
                    tot_out.append(outputs['instances'])

                output_fname = ""
                if args['get'] == 'keypoints':
                    output_fname = './output/joint/detectron/' + file_name + '/' + file_name + '_DJ.pkl'
                    video_writer = cv2.VideoWriter('./output/joint/detectron/' + file_name + '/' + file_name + '_DJ.mp4',
                                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second),
                                               frameSize=(width, height), isColor=True)
                elif args['get'] == 'mask':
                    output_fname = './output/joint/detectron/' + file_name + '/' + file_name + '_DM.pkl'
                    video_writer = cv2.VideoWriter('./output/joint/detectron/' + file_name + '/' + file_name + '_DM.mp4',
                                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second),
                                               frameSize=(width, height), isColor=True)

                with open(output_fname, 'wb') as handle:
                    pickle.dump(tot_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Release resources
                video.release()
                video_writer.release()
                cv2.destroyAllWindows()

                if args['annot_input'] != '-':
                    if os.path.isfile(args['annot_input'] + file_name + '_labels.csv') and os.path.isdir(
                            './output/joint/detectron/' + file_name + '/'):
                        copyfile(args['annot_input'] + file_name + '_labels.csv',
                                 './output/joint/detectron/' + file_name + '/' + file_name + '_labels.csv')
                    else:
                        print('ERROR: missing input/output directory\n' + args[
                            'annot_input'] + file_name + '_labels.csv' + '\n' + './output/joint/detectron/' + file_name + '/')

    if args["method"] == "vibe":

        if not os.path.isdir(args['output_pose_folder'] + args['method'] + '/'):
            os.mkdir(args['output_pose_folder'] + args['method'] + '/')

        subprocess.call(['sh', './main_vibe.sh'])
