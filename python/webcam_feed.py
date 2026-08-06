import cv2

def main():

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the webcam feed.")

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()