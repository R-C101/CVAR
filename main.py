from utils import read_video, save_video
from trackers import Tracker
def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    save_video(video_frames, 'output_videos/outputvideo.avi')
    tracker = Tracker('models/bestest.pt')
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')
    
if __name__ == '__main__':
    main()