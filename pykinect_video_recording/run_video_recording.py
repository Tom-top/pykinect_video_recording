# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:08:11 2021

@author: thomas.topilko
"""

import kinect_video_recording as kvr

experiment, animal = 210319, 125
kinect = kvr.Kinect(experiment, animal)

kinect.set_roi()
kinect.set_16bit_data_range(-1, 750)
kinect.check_depth_histogram(zoom_raw=(400, 800), bins=100) #400, 800

#kinect.check_camera_feed()
#kinect.turn_off()

saving_path = "D://Thomas/Photometry"
#kinect.capture_camera_feed(15,
#                           saving_path,
#                           1 * 60)

kinect.test_clicking_positions()

video_recording_params = {"fps": 10,
                          "recording_duration": 1*60,
                          "inter_recording_duration": 5,
                          "color": True,
                          "depth": True}

kinect.capture_camera_feed_loop(saving_path,
                                **video_recording_params)
