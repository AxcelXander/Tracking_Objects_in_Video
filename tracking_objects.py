# import libraries
import cv2
import numpy as np

#Repeatability
np.random.seed(0)

VfileName = 'video.mp4'
Height = 480
Width = 640


# load video frames from video file

def get_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames

# creating a particle cloud
def initialize_particles(num_particles, frame):
    particles = []
    for i in range(num_particles):
        x = np.random.randint(0, frame.shape[1])
        y = np.random.randint(0, frame.shape[0])
        particles.append([x, y])
    return particles

#moving particles according to the their velocity state
def apply_velocity(particles, velocities):
    for i in range(len(particles)):
        particles[i][0] += velocities[i][0]
        particles[i][1] += velocities[i][1]
    return particles

# prevent particles from failling off the edge of the video frame
def enforce_edges(particles):
    for i in range(len(particles)):
        if particles[i][0] < 0:
            particles[i][0] = 0
        if particles[i][1] < 0:
            particles[i][1] = 0
        if particles[i][0] > Width:
            particles[i][0] = Width
        if particles[i][1] > Height:
            particles[i][1] = Height
    return particles

# measure each particles' quality
def compute_errors(particles,frame):
    errors = []
    for i in range(len(particles)):
        errors.append(compute_error(particles[i],frame))
    return errors

