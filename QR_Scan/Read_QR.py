import cv2
from pyzbar.pyzbar import decode

source = 0 #(camera 0)

def read_qr_code_from_camera():
    # Open de standaard camera (camera 0)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Kan de camera niet openen")
        return

    while True:
        # Lees een frame van de camera
        ret, frame = cap.read()
        if not ret:
            print("Kon geen frame ophalen")
            break

        # Decodeer QR-codes in het frame
        decoded_objects = decode(frame)

        for obj in decoded_objects:
            # Decodeer de QR-code gegevens
            qr_data = obj.data.decode("utf-8")
            print(f"QR Code Data: {qr_data}")

            # Teken een rechthoek rond de gedetecteerde QR-code
            points = obj.polygon
            if len(points) > 4:  # Veronderstel dat het een vierkant is
                hull = cv2.convexHull(points)
                points = hull.reshape(-1, 2)
            
            # Teken de omtrek van de QR-code
            for i in range(len(points)):
                cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 3)

            # Zet de QR-code data op het frame
            x, y, w, h = obj.rect
            cv2.putText(frame, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Toon het frame met de gedetecteerde QR-codes
        cv2.imshow("QR Code Scanner", frame)

        # Druk op 'q' om de loop te verlaten
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ruim de camera en vensters op
    cap.release()
    cv2.destroyAllWindows()

# Roep de functie aan om QR-codes live te scannen
read_qr_code_from_camera()
