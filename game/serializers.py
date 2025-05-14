from rest_framework import serializers
from .models import Word, Target, Score, Player


class WordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Word
        fields = ["text"]


class TargetSerializer(serializers.ModelSerializer):
    word = WordSerializer(read_only=True)

    class Meta:
        model = Target
        fields = ["id", "word"]


class ScoreSubmitSerializer(serializers.Serializer):
    player = serializers.CharField(max_length=64)
    target_id = serializers.IntegerField()
    words = serializers.ListField(
        child=serializers.CharField(max_length=128), min_length=3, max_length=3
    )


class ScoreResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Score
        fields = ["score", "similarities"]
