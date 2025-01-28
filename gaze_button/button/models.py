from django.db import models
from django.utils.timezone import now

class Gaze(models.Model):
    person = models.CharField(max_length=100)
    gaze_direction = models.CharField(max_length=100)
    angle = models.FloatField()
    message = models.CharField(max_length=255, null=True, blank=True)
    timestamp = models.DateTimeField(default=now)
