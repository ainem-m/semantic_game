from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.db import transaction
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import Word, Target, Score, Player
from .serializers import (
    WordSerializer,
    TargetSerializer,
    ScoreSubmitSerializer,
    ScoreResponseSerializer,
)
import umap

client = OpenAI()


def calc_2d(embedding):
    reducer = umap.UMAP(n_components=2, n_neighbors=2, random_state=42)
    # UMAPは2Dでも1サンプルでfit_transformは可能（形だけになるが、例外なく動く）
    coords = reducer.fit_transform(np.array([embedding]))
    return coords[0]


def get_embedding(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return resp.data[0].embedding


def ensure_word(text: str) -> Word:
    # まず存在確認
    try:
        word = Word.objects.get(text=text)
        return word
    except Word.DoesNotExist:
        # 新規単語
        embedding = get_embedding(text)  # OpenAIから取得
        # tsne_x, tsne_y はここでは設定せず、バッチ処理で計算・保存する
        word = Word.objects.create(
            text=text,
            embedding=embedding,
            tsne_x=None,  # 明示的にNoneで初期化
            tsne_y=None,  # 明示的にNoneで初期化
        )
        return word


def calc_score(target_emb, choice_embs):
    sims = sorted(
        [cosine_similarity([target_emb], [e])[0][0] for e in choice_embs],
        reverse=True,
    )
    product = np.prod(sims) * 10000
    return int(round(product)), sims


class WordList(APIView):
    def get(self, request):
        words = Word.objects.filter(
            tsne_x__isnull=False, tsne_y__isnull=False
        )  # 座標があるものだけ
        data = [{"text": w.text, "x": w.tsne_x, "y": w.tsne_y} for w in words]
        return Response(data)


class TargetView(APIView):
    def get(self, request):
        word = Word.objects.order_by("?").first()
        if not word:
            return Response({"detail": "No words in DB"}, status=404)
        target = Target.objects.create(word=word)
        return Response(TargetSerializer(target).data)


class ScoreView(APIView):
    def post(self, request):
        serializer = ScoreSubmitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        # duplicate check
        words_unique = set(data["words"])
        if len(words_unique) < 3:
            return Response({"detail": "Duplicated words"}, status=400)
        # objects
        target = get_object_or_404(Target, id=data["target_id"])
        player, _ = Player.objects.get_or_create(name=data["player"])
        with transaction.atomic():
            target_emb = target.word.embedding or get_embedding(target.word.text)
            choice_words = [ensure_word(w) for w in data["words"]]
            score_int, sims = calc_score(
                target_emb, [w.embedding for w in choice_words]
            )
            score_obj = Score.objects.create(
                player=player,
                target=target,
                choices=[w.text for w in choice_words],
                similarities=sims,
                score=score_int,
            )
        return Response(ScoreResponseSerializer(score_obj).data)
