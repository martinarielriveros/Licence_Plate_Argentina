import cv2

def play_and_resize_video(input_video_path, output_video_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Read the first frame
    ret, frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        print("Error: Unable to read the video file")
        return

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Calculate the scaling factor to fit the frame within the screen dimensions
    scaling_factor = min(1.0, 1200 / width, 800 / height)

    # Calculate the new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Create a VideoWriter object to write the resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (new_width, new_height))

    # Resize the first frame and write it to the output video
    resized_frame = cv2.resize(frame, (new_width, new_height))
    out.write(resized_frame)

    # Continue reading and displaying the video frame by frame
    while cv2.waitKey(1) != ord('q') and ret:  # Press 'q' to quit or end of video
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            cv2.imshow('Resized Video', resized_frame)
            out.write(resized_frame)

    # Release the video capture object, VideoWriter object, and close the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the function to play and resize the video, and store the resized video
input_video_path = './videos/sample1.mp4'
output_video_path = './videos/resized_output.mp4'
play_and_resize_video(input_video_path, output_video_path)

