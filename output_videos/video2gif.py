import cv2
import imageio

video_path = 'output_video7_tracknet4court_yolo4person_unet4ball.avi'
gif_path = video_path[:-4]+'.gif'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)

cap.release()

imageio.mimsave(gif_path, frames, fps=fps)

print(f"GIF saved at {gif_path}")
