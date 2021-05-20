import argparse
import cv2
import os
import pickle

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
    parser.add_argument("--input_video_folder", default='./output/video/',
                        help='specify folder containing video to process')
    parser.add_argument("--output_pose_folder", default='./output/joint/',
                        help='specify folder to save joint output')

    args = vars(parser.parse_args())

    videos = os.listdir(args['input_video_folder'])

    if args["method"] == "detectron":
        if not os.path.isdir(args['output_pose_folder'] + "detectron/"):
            os.mkdir(args['output_pose_folder'] + "detectron/")

        for el in videos:

            if not os.path.isdir(args['output_pose_folder'] + "detectron/" + el):

                os.mkdir("./output/joint/detectron/" + el)
                path = './output/video/' + el
                video = cv2.VideoCapture(path)

                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = video.get(cv2.CAP_PROP_FPS)
                num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                video_writer = cv2.VideoWriter('./output/joint/detectron/' + el + '/' + el + '_D.mp4',
                                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second),
                                               frameSize=(width, height), isColor=True)
                # Initialize predictor
                cfg = get_cfg()  # get a fresh new config
                cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
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

                with open('./output/joint/detectron/' + el + '/' + el + '_DJ.pkl', 'wb') as handle:
                    pickle.dump(tot_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Release resources
                video.release()
                video_writer.release()
                cv2.destroyAllWindows()

    if args["method"] == "vibe":

        if not os.path.isdir(args['output_pose_folder'] + args['method'] + '/'):
            os.mkdir(args['output_pose_folder'] + args['method'] + '/')

        subprocess.call(['sh', './main_vibe.sh'])

