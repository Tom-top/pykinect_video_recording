import sys
from setuptools import setup

PYTHON_VERSION = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)

requirements = [
    'numpy',
    'matplotlib',
    'pyautogui',
    'psutil',
    'datetime',
    'configobj',
    'ffmpeg',
    'opencv',
    'pykinect2',
]

setup(
    name='pykinect_video_recording',
    version='1.0',
    packages=[''],
    install_requires=requirements,
    download_url='git+git://github.com/Tom-top/pykinect_video_recording',
    url='https://github.com/Tom-top/pykinect_video_recording',
    license='GPLv2',
    author='Thomas TOPILKO',
    author_email='thomas.topilko@gmail.com',
    description='video recording tool using PyKinectV2',
    entry_points={
            'console_scripts': ['pykinectvr_show=pykinect_video_recording.video_recording:show_live_feed',
                                'pykinectvr_capture=pykinect_video_recording.video_recording:capture_camera_feed',
                                'pykinectvr_capture_loop=pykinect_video_recording.video_recording:capture_camera_feed_loop',
                                ]},
)
