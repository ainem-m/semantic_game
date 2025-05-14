from django.core.management.base import BaseCommand
from game.models import Word
import numpy as np
import umap


class Command(BaseCommand):
    help = "既存全単語の2次元座標をUMAPで再計算"

    def handle(self, *args, **options):
        words = Word.objects.all()
        if not words:
            self.stdout.write(self.style.WARNING("⚠ 単語がありません"))
            return

        embeddings = np.array([w.embedding for w in words])
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        for w, (x, y) in zip(words, coords):
            w.tsne_x = x  # 気になるなら umap_x にリネーム可
            w.tsne_y = y
            w.save(update_fields=["tsne_x", "tsne_y"])

        self.stdout.write(self.style.SUCCESS("✅ 全単語のUMAP座標更新完了"))
