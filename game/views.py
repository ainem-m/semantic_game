# game/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import ListAPIView  # RankingViewで使用
from django.shortcuts import get_object_or_404
from django.db import transaction
from django.conf import settings  # UMAP_UPDATE_THRESHOLD を読み込むため
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import umap  # ensure_word 内で個別の座標計算はしない方針に変更するなら不要になる可能性

from .models import Word, Target, Score, Player
from .serializers import (
    WordSerializer,  # WordListViewで使う想定（text, x, y を返すように変更が必要）
    TargetSerializer,
    ScoreSubmitSerializer,
    ScoreResponseSerializer,
    PlayerScoreSerializer,  # RankingViewで使用
)

# django-background-tasks のタスクをインポート
try:
    from .tasks import update_all_word_coordinates_dbtask

    BACKGROUND_TASK_LIB_AVAILABLE = True
except ImportError:
    BACKGROUND_TASK_LIB_AVAILABLE = False
    print(
        "Warning: game.tasks.update_all_word_coordinates_dbtask not found. UMAP update trigger will be skipped."
    )

    # ダミー関数でエラーを回避 (実際には log.warning などが望ましい)
    def update_all_word_coordinates_dbtask(*args, **kwargs):
        print(
            "Dummy update_all_word_coordinates_dbtask called because actual task not found."
        )
        pass


client = OpenAI()

# --- Helper Functions ---


def get_embedding(text: str) -> list:  # 型ヒントを list に変更
    resp = client.embeddings.create(
        model="text-embedding-3-large",  # OpenAIのモデル名を確認
        input=text,
    )
    return resp.data[0].embedding


# game/views.py の ensure_word 修正案
def ensure_word(text: str) -> Word:
    """
    指定されたテキストのWordオブジェクトを取得または作成し、embeddingを確実に持つようにする。
    座標(tsne_x, tsne_y)はこの関数では設定せず、バッチ処理に任せる。
    """
    try:
        word = Word.objects.get(text=text)
        if not word.embedding:  # 既存だがembeddingがない場合 (レアケースだが念のため)
            print(f"Word '{text}' found but embedding is missing. Fetching embedding.")
            word.embedding = get_embedding(text)
            # tsne_x, tsne_y は既にNoneのはずなので、ここでは変更しないか、明示的にNoneのままにする
            word.save(update_fields=["embedding"])
    except Word.DoesNotExist:
        print(f"Word '{text}' not found. Creating new entry with embedding.")
        embedding_data = get_embedding(text)
        word = Word.objects.create(
            text=text,
            embedding=embedding_data,
            tsne_x=None,  # 明示的にNoneで初期化
            tsne_y=None,
        )
    return word


def calc_score(target_emb: list, choice_embs: list[list]) -> tuple[int, list]:
    # target_emb, choice_embs がNoneでないことを呼び出し元で保証する想定
    sims = sorted(
        [
            cosine_similarity(
                np.array(target_emb).reshape(1, -1), np.array(e).reshape(1, -1)
            )[0][0]
            for e in choice_embs
        ],
        reverse=True,
    )
    product = np.prod(sims) * 10000
    return int(round(product)), sims


# --- API Views ---


class WordList(APIView):  # フロントエンドのPlotly用
    def get(self, request):
        # 座標が計算済みの単語のみを返す
        words = Word.objects.filter(tsne_x__isnull=False, tsne_y__isnull=False)
        data = [{"text": w.text, "x": w.tsne_x, "y": w.tsne_y} for w in words]
        return Response(data)


class TargetView(APIView):
    def get(self, request):
        # 座標が与えられた (tsne_x が NULL でない) 単語の中からランダムに1つ選択
        word_qs = Word.objects.filter(tsne_x__isnull=False, tsne_y__isnull=False)

        if not word_qs.exists():  # 座標を持つ単語が一つもない場合
            # フォールバック: 全単語からランダムに選択
            word_qs_all = Word.objects.all()
            if not word_qs_all.exists():
                return Response(
                    {"detail": "No words in DB to select as target."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            selected_word = word_qs_all.order_by("?").first()
            print(
                "Warning: No words with coordinates found for target. Selected from all words."
            )
        else:
            selected_word = word_qs.order_by("?").first()

        if not selected_word:  # これも念のため
            return Response(
                {"detail": "Failed to select a target word."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 既存のTargetがあればそれを使うか、毎回新規作成するかは要件次第
        # ここでは毎回新規作成するロジック
        target = Target.objects.create(word=selected_word)
        return Response(TargetSerializer(target).data)


class ScoreView(APIView):
    def post(self, request):
        serializer = ScoreSubmitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # 単語の重複チェック
        words_unique = set(data["words"])
        if len(words_unique) < 3:
            return Response(
                {"detail": "Duplicated words are not allowed."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        target = get_object_or_404(Target, id=data["target_id"])
        player, _ = Player.objects.get_or_create(name=data["player"])

        score_obj_data_for_response = None  # レスポンス用

        with transaction.atomic():
            # ターゲット単語のembeddingを取得 (ensure_wordで確実に取得)
            target_word_obj = ensure_word(
                target.word.text
            )  # target.wordはWordオブジェクトのはず
            target_emb = target_word_obj.embedding

            # 選択された単語のembeddingを取得
            choice_word_objects = [ensure_word(w_text) for w_text in data["words"]]
            choice_embeddings = [w.embedding for w in choice_word_objects]

            # embeddingがNoneの単語がないか最終チェック
            if target_emb is None or any(emb is None for emb in choice_embeddings):
                # ensure_wordで対応されるはずだが、APIエラーなどで取得失敗した場合を考慮
                return Response(
                    {
                        "detail": "Failed to get embeddings for some words. Please try again."
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            score_int, sims = calc_score(target_emb, choice_embeddings)

            score = Score.objects.create(
                player=player,
                target=target,
                choices=[w.text for w in choice_word_objects],  # 保存するのはテキスト
                similarities=sims,
                score=score_int,
            )
            score_obj_data_for_response = ScoreResponseSerializer(score).data

        # --- 座標未計算の単語数をチェックし、閾値を超えたらUMAP更新タスクを起動 ---
        UMAP_UPDATE_THRESHOLD = getattr(settings, "UMAP_UPDATE_THRESHOLD", 10)
        words_without_coords_count = Word.objects.filter(tsne_x__isnull=True).count()

        if words_without_coords_count >= UMAP_UPDATE_THRESHOLD:
            if BACKGROUND_TASK_LIB_AVAILABLE:
                print(
                    f"Found {words_without_coords_count} words without coordinates (threshold: {UMAP_UPDATE_THRESHOLD}). Triggering UMAP update task."
                )
                # タスクに引数が必要なら渡す (この例では不要)
                update_all_word_coordinates_dbtask.now() if hasattr(
                    update_all_word_coordinates_dbtask, "now"
                ) else update_all_word_coordinates_dbtask()
            else:
                print(
                    f"Warning: Found {words_without_coords_count} words without coordinates, but background task library is not available. Skipping UMAP update."
                )
        else:
            if words_without_coords_count > 0:
                print(
                    f"Found {words_without_coords_count} words without coordinates (threshold: {UMAP_UPDATE_THRESHOLD}). UMAP update not triggered yet."
                )

        return Response(score_obj_data_for_response)


class ScoreRankingView(ListAPIView):
    serializer_class = PlayerScoreSerializer

    def get_queryset(self):
        # 単純に全スコアの中から上位3件を取得 (重複プレイヤー許容)
        return Score.objects.select_related("player").order_by("-score")[:3]
        # もし「各プレイヤーの最高スコアTOP3」にするなら、クエリの工夫が必要
        # 例: from django.db.models import Max
        #     top_players = Player.objects.annotate(max_score=Max('score__score')).order_by('-max_score')[:3]
        #     # この後、top_players に紐づく実際のScoreオブジェクトを取得するロジックなど
