import cv2

class camera:
    def __init__(self):
        self.image = None
        self.modele = None

    def CreateModel(self):
        # Capture a frame from the default camera
        self.image = cv2.VideoCapture(0)
        ret, frame = self.image.read()  # Read a frame

        if not ret:
            print("Error: Could not read frame.")
            return

        # Convert the captured frame to grayscale
        self.modele = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("modele.bmp", self.modele)  # Save the model image

        # Release the camera
        self.image.release()

    def SearchModel(self):
        # Check if modele is created
        if self.modele is None:
            print("Error: Model not created. Please call CreateModel() first.")
            return

        # Capture a new frame for searching
        self.image = cv2.VideoCapture(0)
        ret, frame = self.image.read()

        if not ret:
            print("Error: Could not read frame.")
            return

        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Match the template
        res = cv2.matchTemplate(gray_frame, self.modele, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Draw a rectangle around the matched region (optional)
        h, w = self.modele.shape
        cv2.rectangle(frame, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Matched Result', frame)
        cv2.waitKey(0)

        # Release the camera
        self.image.release()
        cv2.destroyAllWindows()

    def ChargeModel(self):
        self.modele = cv2.imread("modele.bmp", cv2.IMREAD_GRAYSCALE)
        if self.modele is None:
            print("Error: Could not load the model image.")