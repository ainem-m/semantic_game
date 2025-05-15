# game/tasks.py

from background_task import background
from django.db import transaction
from django.utils import timezone  # ログ出力用 (任意)
import numpy as np
import umap

from .models import Word  # game.modelsからWordをインポート


@background(
    schedule=0
)  # schedule=0 は「できるだけ早く」実行しようとする (キューに入る)
def update_all_word_coordinates_dbtask():
    """
    全てのWordオブジェクトの2D座標 (tsne_x, tsne_y) をUMAPで再計算し、保存するタスク。
    django-background-tasks によって非同期で実行される。
    """
    task_start_time = timezone.now()
    print(
        f"[{task_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting UMAP coordinate update task..."
    )

    words_to_process = list(Word.objects.all())  # QuerySetをリスト化して処理

    if not words_to_process:
        print("No words found in the database. Task finished.")
        return

    embeddings = []
    valid_words_for_umap = []  # embeddingが有効でUMAP計算の対象となるWordオブジェクトのリスト

    for word_obj in words_to_process:
        if (
            word_obj.embedding
            and isinstance(word_obj.embedding, list)
            and len(word_obj.embedding) > 0
        ):
            embeddings.append(word_obj.embedding)
            valid_words_for_umap.append(word_obj)
        else:
            # embeddingがない、または不正な形式の単語は座標をNULLのままにするか、NULLに設定する
            if word_obj.tsne_x is not None or word_obj.tsne_y is not None:
                word_obj.tsne_x = None
                word_obj.tsne_y = None
                try:
                    word_obj.save(update_fields=["tsne_x", "tsne_y"])
                    print(
                        f"Reset coordinates for word '{word_obj.text}' due to missing/invalid embedding."
                    )
                except Exception as e:
                    print(
                        f"Error resetting coordinates for word '{word_obj.text}': {e}"
                    )

    if not valid_words_for_umap:
        print("No words with valid embeddings found. Task finished.")
        return

    if len(valid_words_for_umap) < 2:
        print(
            f"Only {len(valid_words_for_umap)} word(s) with valid embeddings. UMAP requires at least 2. Setting to (0,0) or None."
        )
        with transaction.atomic():
            for word_obj in words_to_process:  # 全単語を再度ループして処理
                if word_obj in valid_words_for_umap:  # 有効なembeddingを持つ唯一の単語
                    word_obj.tsne_x = 0.0
                    word_obj.tsne_y = 0.0
                else:  # embeddingがなかった単語
                    word_obj.tsne_x = None
                    word_obj.tsne_y = None
                try:
                    word_obj.save(update_fields=["tsne_x", "tsne_y"])
                except Exception as e:
                    print(f"Error saving single/no-coord word '{word_obj.text}': {e}")
        task_end_time_s = timezone.now()
        print(
            f"[{task_end_time_s.strftime('%Y-%m-%d %H:%M:%S')}] UMAP task finished (single/no valid embedding). Duration: {task_end_time_s - task_start_time}"
        )
        return

    # UMAP計算
    # n_neighbors は embedding の数より小さくする必要がある
    num_valid_embeddings = len(embeddings)
    # UMAPのn_neighborsは最小で2。サンプル数が2の場合n_neighborsは1になる。
    n_neighbors_val = (
        min(15, num_valid_embeddings - 1) if num_valid_embeddings > 1 else 1
    )
    if n_neighbors_val < 1:
        n_neighbors_val = 1  # 念のため1未満にならないように

    print(
        f"Calculating UMAP for {num_valid_embeddings} words with n_neighbors={n_neighbors_val}..."
    )

    try:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors_val,
            min_dist=0.1,  # UMAPのパラメータ、適宜調整
            random_state=42,  # 結果の再現性のため
            # low_memory=True # データが大きい場合は検討
        )
        coords_array = reducer.fit_transform(np.array(embeddings))
        print(
            f"UMAP calculation successful. Generated {len(coords_array)} coordinate pairs."
        )
    except Exception as e:
        print(f"Error during UMAP calculation: {e}")
        # エラー発生時は、座標を更新せずにタスクを終了する
        # (または、エラーをログに記録し、部分的にでも処理を続けるか検討)
        return

    # データベース更新
    # valid_words_for_umap と coords_array の要素数は一致するはず
    updated_count = 0
    try:
        with transaction.atomic():  # 複数のsaveをアトミックに
            for i, word_obj in enumerate(valid_words_for_umap):
                if i < len(coords_array):  # 念のため配列の範囲チェック
                    word_obj.tsne_x = float(coords_array[i, 0])
                    word_obj.tsne_y = float(coords_array[i, 1])
                    word_obj.save(update_fields=["tsne_x", "tsne_y"])
                    updated_count += 1
                else:
                    print(
                        f"Warning: Coordinate array shorter than valid words list for word '{word_obj.text}'. Skipping."
                    )

            # UMAP計算の対象にならなかった単語（最初の方で処理済みだが念のため）
            # all_word_ids = {w.id for w in words_to_process}
            # umap_calculated_ids = {w.id for w in valid_words_for_umap}
            # words_without_coords_ids = all_word_ids - umap_calculated_ids
            # if words_without_coords_ids:
            #     Word.objects.filter(id__in=words_without_coords_ids).update(tsne_x=None, tsne_y=None)

        print(f"Successfully updated coordinates for {updated_count} words.")
    except Exception as e:
        print(f"Error during database update with new coordinates: {e}")

    task_end_time = timezone.now()
    print(
        f"[{task_end_time.strftime('%Y-%m-%d %H:%M:%S')}] UMAP coordinate update task finished. Duration: {task_end_time - task_start_time}"
    )
