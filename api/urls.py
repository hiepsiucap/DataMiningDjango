from django.urls import path
from .views import CorrelationAPIView
from .views import AprioriAPIView
from .views import DecisionTreeAPIView
from .views import KMeansClusteringView
from .views import NaiveBayesClassificationView
urlpatterns = [
    path('correlation/', CorrelationAPIView.as_view(), name='correlation'),
    path('apriori/', AprioriAPIView.as_view(), name='apriori'),
    path('decision_tree/', DecisionTreeAPIView.as_view(), name='decision_tree'),
    path('kmeans-clustering/', KMeansClusteringView.as_view(), name='kmeans-clustering'),
    path('naive-bayes/', NaiveBayesClassificationView.as_view(), name='naive_bayes_classification'),
]
