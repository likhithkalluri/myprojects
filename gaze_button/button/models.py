from django.db import models
from django.utils import timezone


class Gaze(models.Model):
    person = models.CharField(max_length=100)
    gaze_direction = models.CharField(max_length=100)
    message = models.CharField(max_length=255, null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    image = models.BinaryField(null=True, blank=True) # Store image as binary data

    def get_image_base64(self):
        """Convert binary image data to Base64 for HTML rendering."""
        import base64
        if self.image:
            return base64.b64encode(self.image).decode('utf-8')
        return None

    def __str__(self):
        return f"{self.person} - {self.gaze_direction} - {self.timestamp}"
