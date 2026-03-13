import subprocess
import sys
import os

def cut_with_ffmpeg(input_video, n_frames, output_video=None):
    if output_video is None:
        base, ext = os.path.splitext(input_video)
        output_video = f"{base}_trimmed{ext}"
    
    # Get video FPS to calculate time to cut
    cmd_probe = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                 '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_video]
    
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    fps_parts = result.stdout.strip().split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) > 1 else float(fps_parts[0])
    
    # Calculate time to cut
    cut_time = n_frames / fps
    
    # Cut video using ffmpeg
    cmd_cut = ['ffmpeg', '-i', input_video, '-ss', str(cut_time), 
               '-c', 'copy', output_video, '-y']
    
    subprocess.run(cmd_cut)
    print(f"Video cut: Removed first {cut_time:.2f} seconds ({n_frames} frames)")

# Usage
if len(sys.argv) >= 3:
    cut_with_ffmpeg(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
