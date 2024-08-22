import cv2
from pyzbar.pyzbar import decode

source = 0  # Camera index (0 for default camera)

def read_qr_code_from_camera():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Cannot open the camera")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Unable to fetch frame")
            break

        # Decode QR codes in the frame
        decoded_objects = decode(frame)

        for obj in decoded_objects:
            # Decode the QR code data
            qr_data = obj.data.decode("utf-8")
            print(f"QR Code Data: {qr_data}")

            # Draw a rectangle around the detected QR code
            points = obj.polygon
            if len(points) > 4:  # Assume it's a polygon if not a rectangle
                hull = cv2.convexHull(points)
                points = hull.reshape(-1, 2)

            # Draw the perimeter of the QR code
            for i in range(len(points)):
                cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 3)

            # Display the QR code data on the frame
            x, y, w, h = obj.rect
            cv2.putText(frame, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the frame with detected QR codes
        cv2.imshow("QR Code Scanner", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start scanning QR codes live
read_qr_code_from_camera()
