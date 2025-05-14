from django.core.management.base import BaseCommand
from game.models import Word
from game.views import ensure_word  # reuse logic


class Command(BaseCommand):
    help = "Generate embeddings for all words without one"

    def handle(self, *args, **kwargs):
        qs = Word.objects.filter(embedding__isnull=True)
        for w in qs:
            ensure_word(w.text)
            self.stdout.write(self.style.SUCCESS(f"embedded {w.text}"))
