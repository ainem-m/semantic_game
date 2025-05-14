from django.db import models


class Word(models.Model):
    text = models.CharField(max_length=128, unique=True)
    embedding = (
        models.JSONField()
    )  # SQLiteならJSONField, PostgreSQLならArrayFieldでも可
    tsne_x = models.FloatField(null=True)
    tsne_y = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Player(models.Model):
    name = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Target(models.Model):
    word = models.ForeignKey(Word, on_delete=models.CASCADE)
    issued_at = models.DateTimeField(auto_now_add=True)


class Score(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    target = models.ForeignKey(Target, on_delete=models.CASCADE)
    choices = models.JSONField()
    similarities = models.JSONField()
    score = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
