# Generated by Django 5.0.3 on 2025-01-28 11:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('button', '0004_remove_gaze_timestamp'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='gaze',
            name='angle',
        ),
    ]
