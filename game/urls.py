from django.urls import path
from .views import WordList, TargetView, ScoreView, ScoreRankingView

urlpatterns = [
    path("words", WordList.as_view()),
    path("target", TargetView.as_view()),
    path("score", ScoreView.as_view()),
    path("ranking", ScoreRankingView.as_view()),
]
