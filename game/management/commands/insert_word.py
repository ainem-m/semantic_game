from django.core.management.base import BaseCommand
from game.models import Word
from django.conf import settings
from openai import OpenAI
from sklearn.manifold import TSNE
import numpy as np
import umap

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def calc_2d(embedding):
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform([embedding])[0]
    return coords


class Command(BaseCommand):
    help = "OpenAI埋め込み取得＋2D座標でWord登録"

    def add_arguments(self, parser):
        parser.add_argument("text", type=str)

    def handle(self, *args, **options):
        text = options["text"]
        if Word.objects.filter(text=text).exists():
            self.stdout.write(self.style.WARNING(f'ℹ "{text}" はすでに存在'))
            return

        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        embedding = resp.data[0].embedding
        x, y = calc_2d(embedding)

        Word.objects.create(text=text, embedding=embedding, tsne_x=x, tsne_y=y)
        self.stdout.write(self.style.SUCCESS(f'✅ "{text}" 登録完了'))
