import subprocess
import sys
import os

def save_first_frames_as_video(input_video, n_frames, output_video=None):
    """
    Save only the first N frames of a video as a new video file.
    
    Args:
        input_video: Path to input video file
        n_frames: Number of first frames to keep
        output_video: Path for output video (optional)
    """
    if output_video is None:
        base, ext = os.path.splitext(input_video)
        output_video = f"{base}_first_{n_frames}_frames{ext}"
    
    # Get video FPS to calculate time duration for N frames
    cmd_probe = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                 '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_video]
    
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    fps_parts = result.stdout.strip().split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) > 1 else float(fps_parts[0])
    
    # Calculate duration for N frames
    duration = n_frames / fps
    
    # Save only the first N frames as a video
    # -t duration specifies how long of the video to take from the start
    cmd_cut = ['ffmpeg', '-i', input_video, 
               '-t', str(duration),  # Take only first 'duration' seconds
               '-c', 'copy',  # Copy codec without re-encoding (fast)
               output_video, '-y']
    
    subprocess.run(cmd_cut)
    print(f"Video saved: First {n_frames} frames ({duration:.2f} seconds)")
    print(f"Output: {output_video}")

# Usage
if len(sys.argv) >= 3:
    save_first_frames_as_video(
        sys.argv[1],  # input video
        int(sys.argv[2]),  # number of frames
        sys.argv[3] if len(sys.argv) > 3 else None  # output video (optional)
    )
