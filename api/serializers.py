from rest_framework import serializers

class ArrayInputSerializer(serializers.Serializer):
    chieucao = serializers.ListField(
        child=serializers.FloatField(), 
        min_length=1, 
        allow_empty=False
    )
    cannang = serializers.ListField(
        child=serializers.FloatField(), 
        min_length=1, 
        allow_empty=False
    )
