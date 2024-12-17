import numpy as np
import cv2

def calculate_distance_traveled(tracked_points, roi_points, field_dimensions, fps=30):
    """
    Calculate the total distance traveled by a player in meters.
    
    Args:
        tracked_points: List of tracked coordinates [(frame, x, y), ...], where (x, y) are the
            player's feet (bottom center of the bounding box)
        roi_points: List of 4 points defining the field corners in image coordinates
        field_dimensions: Tuple of (width, height) in meters
        fps: Frames per second of the video, to average the position every half second
    """
    if not tracked_points or len(tracked_points) < 2:
        return 0

    field_width, field_height = field_dimensions
    
    # Create perspective transform matrix
    roi_points = np.float32(roi_points)
    dst_points = np.float32([
        [0, 0],
        [field_width, 0],
        [field_width, field_height],
        [0, field_height]
    ])
    matrix = cv2.getPerspectiveTransform(roi_points, dst_points)
    
    # Convert tracked points to real-world coordinates, averaged every half second
    window = max(1, int(round(fps / 2)))
    windows = {}
    for frame, x, y in tracked_points:
        point = np.float32([[x, y]])
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix)
        windows.setdefault(frame // window, []).append(transformed.reshape(2))
    real_world_points = [np.mean(windows[w], axis=0) for w in sorted(windows)]
    
    # Calculate total distance
    total_distance = 0
    for i in range(1, len(real_world_points)):
        dx = real_world_points[i][0] - real_world_points[i-1][0]
        dy = real_world_points[i][1] - real_world_points[i-1][1]
        distance = np.sqrt(dx*dx + dy*dy)
        total_distance += distance
    
    return round(total_distance, 2)
