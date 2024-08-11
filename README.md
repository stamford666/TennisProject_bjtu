# TennisProject_bjtu
Tennis analysis system using deep learning and machine learning. <br>
### Tutorials
Prepare a video file with a resolution of 1280x720
1. Clone the repository `https://github.com/stamford666/TennisProject_bjtu.git`
2. Run `pip install -r requirements.txt` to install packages required
3. Run `python main.py <args>`

### Example
0. Raw video
![](pics/video_input.gif)
1. Run `python main.py --court_kps_det_model tracknet --person_det_model yolo --ball_det_model tracknet --draw_trace`
![](pics/output_video7_tracknet4court_yolo4person_tracknet4ball_1.gif)
2. Run `python main.py --court_kps_det_model tracknet --person_det_model yolo --ball_det_model tracknet`
![](pics/output_video7_tracknet4court_yolo4person_tracknet4ball_2.gif)
3. Run `python main.py --court_kps_det_model tracknet --person_det_model fasterrcnn --ball_det_model unet --draw_trace`
![](pics/output_video7_tracknet4court_fasterrcnn4person_unet4ball.gif)


### Reference
https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee
<br>
https://github.com/yastrebksv/TrackNet
<br>
https://arxiv.org/abs/1907.03698
<br>
TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications