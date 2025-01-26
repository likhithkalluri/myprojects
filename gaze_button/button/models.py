from django.db import models
from django.utils import timezone

class GazeRecord(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    direction = models.CharField(max_length=10)
    angle = models.FloatField()
    message = models.TextField()
    screenshot = models.ImageField(upload_to='screenshots/', null=True, blank=True)

    def __str__(self):
        return f"{self.direction} ({self.angle}°) - {self.timestamp}"
