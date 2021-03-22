import kinect_video_recording as kvr
import utilities as utils

if __name__ == "__main__":
    proceed, experiment, tag = utils.setup_recording()
    if proceed:
        config = utils.load_config()
        kinect = kvr.Kinect(experiment, tag)
        kinect.set_roi()
        kinect.set_16bit_data_range(config["general"]["depth_range_min"],
                                    config["general"]["depth_range_max"])
        kinect.capture_camera_feed_loop(config["general"]["saving_dir"],
                                        **config._sections["video_params"])