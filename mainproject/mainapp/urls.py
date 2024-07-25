from django.urls import path
from . import views

urlpatterns = [
        path('',views.crop,name = 'crop'),
        path('index/',views.index,name = 'index'),
        path('advisor/',views.advisor, name = 'advisor'),
        path('newadvisor/',views.newadvisor, name = 'newadvisor'),
        path('result2/',views.result2,name = 'result2'),
        path('about/',views.about,name = "about"),
        path('contact/',views.contact,name = "contact"),
        path('chart/',views.chart,name = 'chart'),
        path('results/',views.results,name = 'results'),
]

from django.conf import settings
from django.conf.urls.static import static
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)