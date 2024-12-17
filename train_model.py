import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import joblib

# Đọc file Excel từ đường dẫn
file_path = '/Users/nguyenhonghiep/Desktop/python/myproject/PlayBall_Data.xlsx'  # Đổi thành đường dẫn file của bạn
sheet_name = 'Sheet1'

# Đọc dữ liệu từ file
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Encode dữ liệu
label_encoders = {}
encoded_df = df.copy()

for column in df.columns:
    le = LabelEncoder()
    encoded_df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Tạo X và y
X = encoded_df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = encoded_df['Play ball']

# Train mô hình
model = CategoricalNB()
model.fit(X, y)

# Lưu mô hình và label encoders
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model và label encoders đã được lưu thành công.")
