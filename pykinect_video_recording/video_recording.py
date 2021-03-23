# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:08:00 2021

@author: thomas.topilko
"""

import os
import time
import shutil
import threading
import queue
from datetime import datetime

import pyautogui
import psutil
import ctypes
import numpy as np
import cv2
import matplotlib.pyplot as plt


from pykinect2 import PyKinectV2, PyKinectRuntime


class KinectException(Exception):
    pass


class Kinect:
    MONITOR_SIZE = pyautogui.size()

    def __init__(self, experiment, animal, verbose=False):
        self._camera_is_on = False
        self._experiment = experiment
        self._animal = animal
        self._verbose = verbose
        self.update_current_time()
        self._position_live_button_doric = (1370, 250)
        self._position_record_button_doric = (1500, 250)

        self.turn_on()
        self._color_width, self._color_height = self.get_color_dimensions()
        self._color_area = self._color_width * self._color_height
        self._color_space_point = PyKinectV2._ColorSpacePoint
        self._window_resizing_factor = 0.3
        self._window_width = int(self._color_width * self._window_resizing_factor)
        self._window_height = int(self._color_height * self._window_resizing_factor)

        self._depth_width, self._depth_height = self.get_depth_dimensions()
        self._depth_area = self._depth_width * self._depth_height
        self._depth_space_point = PyKinectV2._DepthSpacePoint

        self._roi_reference = []
        self.range_for_depth_conversion_is_set = False

        self._video_number = 0
        self._zfilled_video_number = self.get_zfilled_video_number()

    def turn_on(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth |
                                                       PyKinectV2.FrameSourceTypes_Color)
        self._camera_is_on = True
        time.sleep(2)  # This is not a mistake, if this is disabled the kinect
        # Fails to load the first frame.

    def turn_off(self):
        self._kinect.close()
        self._camera_is_on = False

    def get_raw_color_frame(self):
        if not self._camera_is_on:
            self.turn_on()
        if self._kinect.has_new_color_frame():
            color_frame = self._kinect.get_last_color_frame()
            return color_frame
        else:
            if self._verbose:
                print("[ERROR] no new color frame available!")
            return None

    def get_raw_depth_frame(self):
        if not self._camera_is_on:
            self.turn_on()
        if self._kinect.has_new_depth_frame():
            depth_frame = self._kinect.get_last_depth_frame()
            return depth_frame
        else:
            if self._verbose:
                print("[ERROR] no new depth frame available!")
            return None

    def get_color_dimensions(self):
        width = self._kinect.color_frame_desc.Width
        height = self._kinect.color_frame_desc.Height
        return width, height

    def get_depth_dimensions(self):
        width = self._kinect.depth_frame_desc.Width
        height = self._kinect.depth_frame_desc.Height
        return width, height

    def get_color_frame(self, flip=True):
        color_frame = self.get_raw_color_frame()
        if color_frame is None:
            return color_frame
        else:
            color_frame = self.reshape_frame(color_frame,
                                             self._color_width,
                                             self._color_height,
                                             depth=-1)
            if flip:
                color_frame = self.flip_frame_horizontally(color_frame)
            return color_frame

    def get_depth_frame(self, flip=True, return_raw=False):
        raw_depth_frame = self.get_raw_depth_frame()
        if raw_depth_frame is None:
            return raw_depth_frame
        else:
            raw_depth_frame = self.reshape_frame(raw_depth_frame,
                                                 self._depth_width,
                                                 self._depth_height,
                                                 depth=None)
            if flip:
                raw_depth_frame = self.flip_frame_horizontally(raw_depth_frame)
            if return_raw:
                return raw_depth_frame
            else:
                depth_frame = self.frame_to_8bit(raw_depth_frame)
                return depth_frame

    def get_registration_parameters(self):
        """Inspired by https://github.com/KonstantinosAng/
        PyKinect2-Mapper-Functions/blob/master/mapper.py
        """
        color_to_depth_points_type = self._depth_space_point * self._color_area
        color_to_depth_points = ctypes.cast(color_to_depth_points_type(),
                                            ctypes.POINTER(self._depth_space_point))
        self._kinect._mapper. \
            MapColorFrameToDepthSpace(ctypes.c_uint(self._depth_area),
                                      self._kinect._depth_frame_data,
                                      ctypes.c_uint(self._color_area),
                                      color_to_depth_points)
        depth_x_y = np.copy(np.ctypeslib.as_array(color_to_depth_points,
                                                  shape=(self._color_area,)))
        depth_x_y = depth_x_y.view(np.float32).reshape(depth_x_y.shape + (-1,))
        depth_x_y += 0.5
        depth_x_y = depth_x_y.reshape(self._color_height,
                                      self._color_width,
                                      2).astype(int)
        depth_x_params = np.clip(depth_x_y[:, :, 0], 0, self._depth_width - 1)
        depth_y_params = np.clip(depth_x_y[:, :, 1], 0, self._depth_height - 1)

        return depth_x_params, depth_y_params

    def register_depth_to_color(self, flip=True):
        raw_depth_frame = self.get_depth_frame(flip=False,
                                               return_raw=True)  # The flip=False is mandatory for the registration
        if raw_depth_frame is None:
            return None
        registered_depth_frame = np.zeros((self._color_height, self._color_width),
                                          dtype=np.uint16)
        depth_x_params, depth_y_params = self.get_registration_parameters()
        registered_depth_frame[:, :] = raw_depth_frame[depth_y_params, depth_x_params]
        if flip:
            registered_depth_frame = self.flip_frame_horizontally(registered_depth_frame)
        return registered_depth_frame

    def set_cropping_params(self, upper_left, lower_right):
        self._roi_reference = [upper_left, lower_right]
        self._roi_width = abs(self._roi_reference[0][0] - self._roi_reference[1][0])
        self._roi_height = abs(self._roi_reference[0][1] - self._roi_reference[1][1])

    def set_roi(self, resize_window=None):
        self._roi_reference = []
        self.color_frame_roi = self.get_color_frame()
        self.color_clone_roi = self.color_frame_roi.copy()

        cv2.namedWindow("color_frame", cv2.WINDOW_NORMAL)
        if resize_window is None:
            cv2.resizeWindow("color_frame", (self._window_width, self._window_height))
        else:
            window_width = int(self._color_width * resize_window)
            window_height = int(self._color_height * resize_window)
            cv2.resizeWindow("color_frame", (window_width, window_height))
        cv2.setMouseCallback("color_frame", self.select_and_crop)

        while True:
            cv2.imshow("color_frame", self.color_frame_roi)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("r"):
                if self._verbose:
                    print("[INFO] ROI was reset")
                self._roi_reference = []
                self.color_frame_roi = self.color_clone_roi.copy()
            elif key == ord("c"):
                if self._verbose:
                    print("[INFO] ROI successfully set")
                break

        cv2.destroyAllWindows()

        for i in range(1, 5):
            cv2.waitKey(1)

    def select_and_crop(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._roi_reference = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self._roi_reference.append((x, y))

        if len(self._roi_reference) == 2:
            self._roi_width = abs(self._roi_reference[0][0] - self._roi_reference[1][0])
            self._roi_height = abs(self._roi_reference[0][1] - self._roi_reference[1][1])
            cv2.rectangle(self.color_frame_roi,
                          self._roi_reference[0],
                          self._roi_reference[1],
                          (0, 0, 255),
                          2)
            cv2.imshow("color_frame", self.color_frame_roi)

    def crop_frame_from_params(self, frame):
        if self._roi_reference:
            return frame[self._roi_reference[0][1]:self._roi_reference[1][1],
                         self._roi_reference[0][0]:self._roi_reference[1][0]]
        else:
            return frame

    def set_16bit_data_range(self, vmin=0, vmax=256):
        vmin, vmax = int(vmin), int(vmax)
        if vmin < 0 and vmax < 0:
            raise KinectException("vmin and vmax can't both be negative")
        if vmin > vmax:
            raise KinectException("vmin has to be superior to vmax!")
        if vmin == -1:
            vmin = vmax - 256
        if vmax == -1:
            vmax = vmin + 256
        if vmax - vmin < 256:
            print("[WARNING] vmax-vmin is superior to 256 clipping the range automatically")
            vmin = vmax - 256
        self._depth_vmin = vmin
        self._depth_vmax = vmax
        self.range_for_depth_conversion_is_set = True

    def check_depth_histogram(self, zoom_raw=(), bins=300):
        depth_frame = self.register_depth_to_color()
        cropped_depth_frame = self.crop_frame_from_params(depth_frame)
        scaled_depth_frame = self.scale_image_to_8bit_from_data_range(cropped_depth_frame,
                                                                      self._depth_vmin,
                                                                      self._depth_vmax)

        plt.figure(figsize=(4, 7))
        y_max = 10000

        ax0 = plt.subplot(4, 1, 1)
        if zoom_raw:
            ax0.imshow(cropped_depth_frame, vmin=zoom_raw[0], vmax=zoom_raw[-1], aspect='auto')
        else:
            ax0.imshow(cropped_depth_frame, vmin=0, vmax=2 ** 16, aspect='auto')
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = plt.subplot(4, 1, 2)
        ax1.hist(cropped_depth_frame.flatten(), bins=bins)
        ax1.vlines([self._depth_vmin, self._depth_vmax],
                   ymin=0,
                   ymax=y_max,
                   color="red",
                   linewidth=1)
        if zoom_raw:
            ax1.set_xlim(zoom_raw[0], zoom_raw[1])
        else:
            ax1.set_xlim(0, 2 ** 16)
        ax1.set_ylim(0, y_max)

        ax2 = plt.subplot(4, 1, 3)
        ax2.imshow(scaled_depth_frame, vmin=0, vmax=2 ** 8, aspect='auto')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = plt.subplot(4, 1, 4)
        ax3.hist(scaled_depth_frame.flatten(), bins=bins)
        ax3.set_xlim(0, 2 ** 8)
        ax3.set_ylim(0, y_max)

        plt.tight_layout()
        plt.show()

    def set_window_resizing_factor(self, percentage):
        self._window_resizing_factor = percentage

    def show_live_feed(self, color=True, depth=True, flip=True):
        if color and not depth:
            cv2.namedWindow("color", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("color", (self._window_width, self._window_height))
        elif depth and not color:
            cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("depth", (self._window_width, self._window_height))
        elif color and depth:
            cv2.namedWindow("color_and_depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("color_and_depth", (self._window_width * 2, self._window_height))

        while True:
            if color:
                current_color_frame = self.get_color_frame(flip=flip)
                current_color_frame = self.crop_frame_from_params(current_color_frame)
                current_color_frame = current_color_frame[:, :, :3]
            if depth:
                current_depth_frame = self.register_depth_to_color(flip=flip)
                current_depth_frame = self.crop_frame_from_params(current_depth_frame)
                if self.range_for_depth_conversion_is_set:
                    current_depth_frame = self.scale_image_to_8bit_from_data_range(current_depth_frame,
                                                                                   self._depth_vmin,
                                                                                   self._depth_vmax)
                else:
                    current_depth_frame = self.scale_image_to_8bit(current_depth_frame)

            if color and not depth:
                cv2.imshow("color", current_color_frame)
            elif depth and not color:
                cv2.imshow("depth", current_depth_frame)
            elif color and depth:
                color_and_depth = cv2.hstack((current_color_frame, current_depth_frame))
                cv2.imshow("color_and_depth", color_and_depth)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def set_video_writers(self, fps, saving_directory, color=True, depth=True):
        self.update_current_time()
        if color:
            color_video_path = os.path.join(saving_directory,
                                            "color_video_{}_{}_{}.mp4"
                                            .format(self._experiment,
                                                    self._zfilled_video_number,
                                                    self._animal))
            if self._roi_reference:
                color_width, color_height = self._roi_width, self._roi_height
            else:
                color_width, color_height = self._color_width, self._color_height
            self.video_writer_color = cv2.VideoWriter(color_video_path,
                                                      cv2.VideoWriter_fourcc(*"DIVX"),
                                                      fps,
                                                      (color_width,
                                                       color_height))
        if depth:
            depth_video_path = os.path.join(saving_directory,
                                            "depth_video_{}_{}_{}.mp4"
                                            .format(self._experiment,
                                                    self._zfilled_video_number,
                                                    self._animal))
            if self._roi_reference:
                depth_width, depth_height = self._roi_width, self._roi_height
            else:
                depth_width, depth_height = self._depth_width, self._depth_height
            self.video_writer_depth = cv2.VideoWriter(depth_video_path,
                                                      cv2.VideoWriter_fourcc(*"DIVX"),
                                                      fps,
                                                      (depth_width,
                                                       depth_height),
                                                      False)

    def capture_camera_feed(self,
                            saving_directory,
                            fps=15,
                            recording_duration=20 * 60,
                            color=True,
                            depth=True):
        if not any([color, depth]):
            raise ValueError("Both color and depth were set to False!\n\
                             at least one camera feed has to be enabled")
        self.stop_capture = False
        self.set_video_writers(fps, saving_directory, color=color, depth=depth)
        self.queue_video_writer_color = VideoWriter(self.video_writer_color,
                                                    saving_dir=saving_directory,
                                                    fn="color",
                                                    fps=fps,
                                                    duration=recording_duration)
        self.queue_video_writer_depth = VideoWriter(self.video_writer_depth,
                                                    saving_dir=saving_directory,
                                                    fn="depth",
                                                    fps=fps,
                                                    duration=recording_duration)
        delay_between_frames = 1 / fps
        start_timer = time.time()
        fps_timer = time.time()
        self.move_to_and_click(self._position_record_button_doric)

        while True:
            if self.stop_capture:
                break
            cur_timer = time.time()
            if (cur_timer - fps_timer) >= delay_between_frames:
                if self._verbose:
                    percentage_error = (((cur_timer - fps_timer) -
                                         delay_between_frames) /
                                        delay_between_frames) * 100
                    print(psutil.virtual_memory().percent)
                    print(percentage_error)
                if color:
                    color_frame = None
                    while color_frame is None:
                        color_frame = self.get_color_frame(flip=True)
                    color_frame = self.crop_frame_from_params(color_frame)
                    color_frame = color_frame[:, :, :3]
                    self.queue_video_writer_color.write(color_frame)
                if depth:
                    depth_frame = None
                    while depth_frame is None:
                        depth_frame = self.register_depth_to_color(flip=True)
                    depth_frame = self.crop_frame_from_params(depth_frame)
                    if self.range_for_depth_conversion_is_set:
                        depth_frame = self.scale_image_to_8bit_from_data_range(depth_frame,
                                                                               self._depth_vmin,
                                                                               self._depth_vmax)
                    depth_frame = depth_frame.astype("uint8")
                    self.queue_video_writer_depth.write(depth_frame)
                fps_timer = cur_timer
            if (cur_timer - start_timer) > recording_duration:
                self.move_to_and_click(self._position_record_button_doric)
                break

    def capture_camera_feed_loop(self,
                                 saving_directory,
                                 fps=15,
                                 recording_duration=20 * 60,
                                 inter_recording_duration=20 * 60,
                                 color=True,
                                 depth=True):
        shutil.copy(os.path.join(os.getcwd(), "config.cfg"),
                    os.path.join(saving_directory, "config_{}_{}.ini" \
                                 .format(self._animal, self._experiment)),)
        # fps = int(fps)
        # recording_duration = int(recording_duration)
        # inter_recording_duration = int(inter_recording_duration)
        # color = bool(color)
        # depth = bool(depth)
        can_start = True
        while True:
            try:
                if can_start:
                    self._zfilled_video_number = self.get_zfilled_video_number()
                    saving_folder_path = os.path.join(saving_directory,
                                                      "{}_{}".format(self._animal,
                                                                     self._zfilled_video_number))
                    if os.path.exists(saving_folder_path):
                        if os.listdir(saving_folder_path) != 0:
                            self.clean_directory(saving_folder_path)
                    os.mkdir(saving_folder_path)
                    self.capture_camera_feed(saving_folder_path,
                                             fps=fps,
                                             recording_duration=recording_duration,
                                             inter_recording_duration=inter_recording_duration,
                                             color=color,
                                             depth=depth)
                    self._video_number += 1
                    can_start = False
                    start_timer = time.time()
                else:
                    current_time = time.time()
                    delta = current_time - start_timer
                    if delta > inter_recording_duration:
                        can_start = True
            except KeyboardInterrupt:
                self.stop_capture = True
                self.queue_video_writer_color.stop_and_release()
                self.queue_video_writer_depth.stop_and_release()
                break

    def get_zfilled_video_number(self, n_zeros=4):
        return str(self._video_number).zfill(n_zeros)

    def test_clicking_positions(self):
        pyautogui.moveTo(self._position_live_button_doric)
        pyautogui.moveTo(self._position_record_button_doric)

    def update_current_time(self):
        self._time = str(datetime.now().strftime("%H-%M-%S"))

    @staticmethod
    def frame_to_8bit(frame):
        return frame.astype(np.uint8)

    @staticmethod
    def resize_frame(frame, resizing_factor):
        pass

    @staticmethod
    def reshape_frame(frame, width, height, depth=None):
        if depth is not None:
            return frame.reshape(height, width, depth)
        else:
            return frame.reshape(height, width)

    @staticmethod
    def flip_frame_horizontally(frame):
        return cv2.flip(frame, 1)

    @staticmethod
    def flip_frame_vertically(frame):
        return cv2.flip(frame, 0)

    @staticmethod
    def frame_bgr_to_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def frame_rgb_to_bgr(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def blend_frames(frame1, frame2):
        return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0.0)

    @staticmethod
    def show_frame(frame, range_im_1=(None, None)):
        plt.figure(figsize=(5, 5))
        ax0 = plt.subplot(1, 1, 1)
        ax0.imshow(frame, vmin=range_im_1[0], vmax=range_im_1[1])

    @staticmethod
    def show_two_frames(frame1, frame2, range_im_1=(None, None), range_im_2=(None, None)):
        plt.figure(figsize=(15, 5))
        ax0 = plt.subplot(1, 2, 1)
        ax0.imshow(frame1, vmin=range_im_1[0], vmax=range_im_1[1])
        ax1 = plt.subplot(1, 2, 2)
        ax1.imshow(frame2, vmin=range_im_2[0], vmax=range_im_2[1])

    @staticmethod
    def show_histogram_pixel_intensity(frame, bins=200):
        if len(frame.shape) > 1:
            frame = frame.flatten()
        plt.figure(figsize=(5, 5))
        ax0 = plt.subplot(1, 1, 1)
        ax0.hist(frame, bins=bins)

    @staticmethod
    def threshold_image(frame, low, high):
        frame[frame <= low] = low
        frame[frame >= high] = high
        return frame

    @staticmethod
    def scale_image_to_8bit(frame):
        return ((frame - min(frame)) / (max(frame) - min(frame))) * 256

    @staticmethod
    def scale_image_to_8bit_from_data_range(frame, vmin, vmax):
        frame_copy = frame.copy()
        frame_copy[frame_copy < vmin] = vmin
        frame_copy[frame_copy > vmax] = vmax
        scaled_frame = ((frame_copy - vmin) / (vmax - vmin)) * 256
        scaled_frame = scaled_frame.astype("uint8")
        return scaled_frame

    @staticmethod
    def crop_frame(frame, upper_left, lower_right):
        y0, x0 = upper_left
        y1, x1 = lower_right
        return frame[x0:x1, y0:y1]

    @staticmethod
    def divide_until_smaller(source, target):
        res = 1
        while source > target:
            source = source / 2
            res *= 2
        return res

    @staticmethod
    def move_to_and_click(button):
        pyautogui.moveTo(button)
        pyautogui.click()

    @staticmethod
    def clean_directory(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


class VideoWriter:
    def __init__(self, video_writer, fn=0, duration=20 * 60, fps=15):
        self._video_duration = duration
        self._video_fps = fps
        self._dt = 1. / self._video_fps
        self._video_writer = video_writer
        self._queue = queue.Queue()
        self._stop = False
        self._n = 0
        self._fn = fn
        self._wrtr = threading.Thread(target=self.queue_writer)
        self._wrtr.start()

    def queue_writer(self):
        current_frame = None
        start_time = None
        while True:
            if self._stop:
                break
            while not self._queue.empty():
                current_frame = self._queue.get_nowait()
            if current_frame is not None:
                if start_time is None:
                    start_time = time.time()
                self._video_writer.write(current_frame)
                self._n += 1
                if self._n == self._video_duration * self._video_fps:
                    self.stop_and_release()
            dt = self._dt if start_time is None else \
                max(0, start_time + self._n * self._dt - time.time())
            time.sleep(dt)

    def write(self, frm):
        self._queue.put(frm)

    def release(self):
        self._video_writer.release()

    def stop(self):
        self._stop = True

    def stop_and_release(self):
        self.stop()
        self.release()
