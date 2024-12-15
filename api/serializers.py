from rest_framework import serializers

class ArrayInputSerializer(serializers.Serializer):
    input = serializers.ListField(
        child=serializers.IntegerField(),  # Nếu là số nguyên
        allow_empty=False
    )
