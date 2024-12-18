import math
import matplotlib
matplotlib.use('Agg')
from django.core.files.base import ContentFile
from sklearn.cluster import KMeans
from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
from rest_framework import status
from .serializers import ArrayInputSerializer
from rest_framework.parsers import MultiPartParser,FormParser
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from django.core.files.storage import default_storage
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import os
import cloudinary.uploader
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.views import View
import os
import uuid
from rest_framework import serializers


class CorrelationAPIView(APIView):
    def post(self, request):
        # Áp dụng serializer cho dữ liệu đầu vào
        serializer = ArrayInputSerializer(data=request.data)
        if serializer.is_valid():
            chieucao = serializer.validated_data['chieucao']
            cannang = serializer.validated_data['cannang']
            
            # Kiểm tra xem độ dài 2 mảng có khớp nhau không
            if len(chieucao) != len(cannang):
                return Response(
                    {"error": "chieucao và cannang phải có cùng độ dài."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Tính phương sai o2x và o2y
            tempA = tempB = 0
            for i in chieucao:
                tempA += i
                tempB += i * i
            o2x = tempB / len(chieucao) - (tempA / len(chieucao)) ** 2

            tempA = tempB = 0
            for j in cannang:
                tempA += j
                tempB += j * j
            o2y = tempB / len(cannang) - (tempA / len(cannang)) ** 2

            # Tính covariance và b1
            tempA = tempB = tempC = 0
            for i in range(0, len(chieucao)):
                tempA += chieucao[i]
                tempB += cannang[i]
                tempC += chieucao[i] * cannang[i]

            tempA /= len(chieucao)
            tempB /= len(cannang)
            tempC /= len(chieucao)

            b1 = (tempC - tempA * tempB) / o2x
            r = b1 * (math.sqrt(o2x)) / math.sqrt(o2y)
            rounded_r = round(r, 2)

            # Trả về kết quả
            return Response(
                {"correlation": rounded_r},
                status=status.HTTP_200_OK
            )
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class AprioriAPIView(APIView):
    parser_classes = [MultiPartParser]  # Xử lý file upload

    def post(self, request):
        try:
            file = request.FILES['file']  
            min_support = float(request.data.get('min_support', 0.5))
            min_confidence = float(request.data.get('min_confidence', 0.5))
            df = pd.read_excel(file)
            pivot_df = df.pivot_table(index='Mã hoá đơn', columns='Mặt hàng', aggfunc='size', fill_value=0)
            pivot_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)
            frequent_itemsets = apriori(pivot_df, min_support=min_support, use_colnames=True)

            # Hàm kiểm tra tập phổ biến tối đại
            def is_maximal(frequent_itemsets):
                maximal_itemsets = []
                for i, itemset in enumerate(frequent_itemsets['itemsets']):
                    is_subset = False
                    for j, other_itemset in enumerate(frequent_itemsets['itemsets']):
                        if i != j and set(itemset).issubset(other_itemset):
                            is_subset = True
                            break
                    if not is_subset:
                        maximal_itemsets.append(itemset)
                return maximal_itemsets

            maximal_itemsets = is_maximal(frequent_itemsets)

            # Tính luật kết hợp
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

            # Trả kết quả
            response_data = {
                'frequent_itemsets': frequent_itemsets.to_dict(orient='records'),
                'maximal_itemsets': [list(item) for item in maximal_itemsets],
                'rules': rules[['antecedents', 'consequents', 'confidence']].to_dict(orient='records')
            }

            return Response(response_data, status=200)

        except Exception as e:
            return Response({'error': str(e)}, status=400)

from django.conf import settings

class DecisionTreeAPIView(View):
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        try:
            # Save the uploaded file temporarily
            temp_file = f"{uuid.uuid4().hex}.xlsx"
            file_path = default_storage.save(temp_file, file)

            # Process the Excel file
            data = pd.read_excel(file_path)
            data = data.drop(columns=['Day'])  # Drop the 'Day' column if it exists
            data = pd.get_dummies(data, drop_first=True)

            # Prepare training and testing data
            X = data.drop(columns=['Play?_Yes'])
            y = data['Play?_Yes']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train Decision Tree models
            clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
            clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
            clf_gini.fit(X_train, y_train)
            clf_entropy.fit(X_train, y_train)

            # Temporary local file paths for plots
            gini_tree_path = os.path.join(settings.MEDIA_ROOT, f"gini_tree_{uuid.uuid4().hex}.png")
            entropy_tree_path = os.path.join(settings.MEDIA_ROOT, f"entropy_tree_{uuid.uuid4().hex}.png")

            # Plot the decision trees
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=["No", "Yes"], ax=axes[0])
            axes[0].set_title("Decision Tree (Gini Index)")
            plot_tree(clf_entropy, filled=True, feature_names=X.columns, class_names=["No", "Yes"], ax=axes[1])
            axes[1].set_title("Decision Tree (Information Gain)")
            fig.savefig(gini_tree_path)
            fig.savefig(entropy_tree_path)
            plt.close()

            # Upload plots to Cloudinary
            gini_result = cloudinary.uploader.upload(gini_tree_path, folder="decision_trees")
            entropy_result = cloudinary.uploader.upload(entropy_tree_path, folder="decision_trees")

            # Delete temporary files
            os.remove(gini_tree_path)
            os.remove(entropy_tree_path)
            default_storage.delete(file_path)

            # Return Cloudinary URLs
            return JsonResponse({
                "gini_tree": gini_result['secure_url'],
                "entropy_tree": entropy_result['secure_url']
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    def get(self, request):
        return JsonResponse({"message": "Please use POST method to upload a file."}, status=405)
class KMeansClusteringView(APIView):
    def post(self, request):
        try:
            # Check if file is uploaded
            if 'file' not in request.FILES:
                return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Save uploaded file
            file = request.FILES['file']
            file_path = default_storage.save('uploaded_files/' + file.name, ContentFile(file.read()))
            
            # Read Excel file
            full_path = os.path.join(default_storage.location, file_path)
            data = pd.read_excel(full_path)
            
            # Validate data columns
            if 'X' not in data.columns or 'Y' not in data.columns:
                return Response({'error': 'File must contain X and Y columns'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Prepare data for K-Means
            X = data[['X', 'Y']].values
            
            # Get number of clusters from request (default to 3)
            n_clusters = request.data.get('n_clusters', 3)
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=int(n_clusters), random_state=42)
            kmeans.fit(X)
            
            # Prepare response data
            labels = kmeans.labels_.tolist()
            centroids = kmeans.cluster_centers_.tolist()
            
            # Clean up uploaded file
            default_storage.delete(file_path)
            
            return Response({
                'labels': labels,
                'centroids': centroids,
                'points': X.tolist()
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
import joblib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load model và encoders
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
ENCODER_PATH= os.path.join(BASE_DIR, 'label_encoders.pkl')

class PredictView(APIView):
    def post(self, request):
        try:
            # Tải mô hình và label encoders
            model = joblib.load(MODEL_PATH)
            label_encoders = joblib.load(ENCODER_PATH)

            # Lấy dữ liệu từ request
            input_data = request.data
            sample = pd.DataFrame([{
                'Outlook': input_data['Outlook'],
                'Temperature': input_data['Temperature'],
                'Humidity': input_data['Humidity'],
                'Wind': input_data['Wind']
            }])

            # Encode dữ liệu đầu vào
            for column in sample.columns:
                if column in label_encoders:
                    sample[column] = label_encoders[column].transform(sample[column])
                else:
                    return Response({'error': f'Invalid column: {column}'}, status=status.HTTP_400_BAD_REQUEST)

            # Dự đoán
            prediction = model.predict(sample)
            predicted_label = label_encoders['Play ball'].inverse_transform(prediction)[0]

            return Response({'prediction': predicted_label}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class DataProcessingAPIView(APIView):
    parser_classes = [MultiPartParser]  # Handles file uploads

    def post(self, request):
        try:
            # Receive file
            file = request.FILES['file']
            df = pd.read_excel(file)

            # Receive parameters
            decision_attribute = request.data.get('decision_attribute')  # Decision attribute
            X_indices = set(map(int, request.data.get('X', '0,2,3').split(',')))  # X indices
            condition_attributes = [attr for attr in request.data.get('condition_attributes', '').split(',') if attr]

            # Validate inputs
            if decision_attribute not in df.columns:
                raise ValueError(f"Decision attribute '{decision_attribute}' not found in dataset.")
            for attr in condition_attributes:
                if attr and attr not in df.columns:
                    raise ValueError(f"Condition attribute '{attr}' not found in dataset.")
            if not all(0 <= idx < len(df) for idx in X_indices):
                raise ValueError("Some indices in 'X' are out of range.")

            # Equivalence classes
            equivalence = self.compute_equivalence_classes(df, condition_attributes)

            # Approximation calculations
            universe = set(df.index)
            lower_approx, upper_approx = self.compute_approximations(universe, X_indices, equivalence)
            boundary_region = upper_approx - lower_approx
            outside_region = universe - upper_approx

            # Dependency calculation
            dependency = self.calculate_dependency(df, decision_attribute, condition_attributes)

            # Reduct computation
            reduct_attributes = self.compute_reduct(df, decision_attribute)

            # Result
            result = {
                'equivalence_classes': {str(k): v for k, v in equivalence.items()},  # Convert keys to strings
                'lower_approximation': list(lower_approx),
                'upper_approximation': list(upper_approx),
                'boundary_region': list(boundary_region),
                'outside_region': list(outside_region),
                'dependency_degree': dependency,
                'reduct_attributes': reduct_attributes,
            }

            return Response(result, status=200)

        except Exception as e:
            return Response({'error': str(e)}, status=400)

    def compute_equivalence_classes(self, data, attributes):
        equivalence = {}
        for index, row in data.iterrows():
            key = tuple(row[attributes])
            str_key = str(key)  # Convert tuple to string
            equivalence.setdefault(str_key, []).append(index)  # Use string as key
        return equivalence

    def compute_approximations(self, universe, target_class, equivalence):
        lower_approx, upper_approx = set(), set()
        for eq_class in equivalence.values():
            eq_set = set(eq_class)
            if eq_set.issubset(target_class):
                lower_approx.update(eq_set)
            if eq_set & target_class:
                upper_approx.update(eq_set)
        return lower_approx, upper_approx

    def calculate_dependency(self, data, decision_attribute, condition_attributes):
        total_rows = len(data)
        equivalence_classes = self.compute_equivalence_classes(data, condition_attributes)
        dependent_rows = 0

        for eq_class in equivalence_classes.values():
            decision_values = data.loc[eq_class, decision_attribute].unique()
            if len(decision_values) == 1:
                dependent_rows += len(eq_class)

        return dependent_rows / total_rows

    def compute_reduct(self, data, decision_attribute):
        all_attributes = [col for col in data.columns if col != decision_attribute]
        full_diff_matrix = self.compute_difference_matrix(data, all_attributes)

        reduct = set(all_attributes)
        for attr in all_attributes:
            remaining_attrs = [a for a in all_attributes if a != attr]
            reduced_matrix = self.compute_difference_matrix(data, remaining_attrs)

            if np.array_equal(full_diff_matrix, reduced_matrix):
                reduct.remove(attr)

        return list(reduct)

    def compute_difference_matrix(self, data, attributes):
        equivalence = self.compute_equivalence_classes(data, attributes)
        n = len(data)
        diff_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                diff_matrix[i][j] = not data.iloc[i][attributes].equals(data.iloc[j][attributes])
        return diff_matrix
