import numpy as np
import cv2

def calculate_distance_traveled(tracked_points, roi_points, field_dimensions):
    """
    Calculate the total distance traveled by a player in meters.
    
    Args:
        tracked_points: List of tracked coordinates [(frame, x, y), ...]
        roi_points: List of 4 points defining the field corners in image coordinates
        field_dimensions: Tuple of (width, height) in meters
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
    
    # Convert tracked points to real-world coordinates
    real_world_points = []
    for frame, x, y in tracked_points:
        point = np.float32([[x, y]])
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix)
        real_world_points.append(transformed.reshape(2))
    
    # Calculate total distance
    total_distance = 0
    for i in range(1, len(real_world_points)):
        dx = real_world_points[i][0] - real_world_points[i-1][0]
        dy = real_world_points[i][1] - real_world_points[i-1][1]
        distance = np.sqrt(dx*dx + dy*dy)
        total_distance += distance
    
    return round(total_distance, 2)
