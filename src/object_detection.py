class ObjectDetectionResult:
    """
    Stores the detected object's location in an image.
    """

    def __init__(self, object_name, x1, y1, x2, y2, confidence=1.0):
        self.object_name = object_name

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.confidence = confidence

    @property
    def center(self):
        """Return the center of the bounding box."""
        x = (self.x1 + self.x2) / 2
        y = (self.y1 + self.y2) / 2

        return x, y

    @property
    def ground_contact_point(self):
        """
        Return the bottom-center point of the object.

        This is more useful for ground-plane distance
        calculations than the bounding-box center.
        """

        x = (self.x1 + self.x2) / 2
        y = self.y2

        return x, y


def detection_status():
    print("Object detection module ready.")
    print("Waiting for CCTV footage.")


if __name__ == "__main__":
    detection_status()