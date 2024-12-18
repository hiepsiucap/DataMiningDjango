from django.urls import path
from .views import CorrelationAPIView
from .views import AprioriAPIView
from .views import DecisionTreeAPIView
from .views import KMeansClusteringView
from .views import PredictView
from .views import DataProcessingAPIView
urlpatterns = [
    path('correlation/', CorrelationAPIView.as_view(), name='correlation'),
    path('apriori/', AprioriAPIView.as_view(), name='apriori'),
    path('process_data/', DataProcessingAPIView.as_view(), name='process_data'),
    path('decision_tree/', DecisionTreeAPIView.as_view(), name='decision_tree'),
    path('kmeans-clustering/', KMeansClusteringView.as_view(), name='kmeans-clustering'),
    path('naive-bayes/', PredictView.as_view(), name='naive_bayes_classification'),
]
