from django.db import models

class Gaze(models.Model):
    person = models.CharField(max_length=100)
    gaze_direction = models.CharField(max_length=100)
    message = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"{self.person}: {self.gaze_direction} ({self.angle}°) at {self.timestamp}"
