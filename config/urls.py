from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("game.urls")),
    path("", include("game.urls_html")),  # html endpoints
]
